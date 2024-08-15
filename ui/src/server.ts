import {
  split,
  HttpLink,
  ApolloClient,
  InMemoryCache,
  gql,
} from "@apollo/client/core";
import { getMainDefinition } from "@apollo/client/utilities";
import { GraphQLWsLink } from "@apollo/client/link/subscriptions";
import { createClient } from "graphql-ws";

const httpLink = new HttpLink({
  uri: "http://localhost:8000/graphql",
});

const wsLink = new GraphQLWsLink(
  createClient({
    url: "ws://localhost:8000/graphql",
  })
);

const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === "OperationDefinition" &&
      definition.operation === "subscription"
    );
  },
  wsLink,
  httpLink
);

const client = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
});

export const LOAD_MODEL_QUERY = gql`
  query LoadModel {
    loadModel {
      message
    }
  }
`;

export const GENERATIONS_QUERY = gql`
  query Generations {
    generations {
      images
    }
  }
`;

export const GENERATION_QUERY = gql`
  query Generation($id: String!) {
    generation(id: $id) {
      images
      step
      totalSteps
    }
  }
`;

export type Generation = {
  id: string;
  images: string[];
  step: number;
  totalSteps: number;
};

export const GENERATION_SUBSCRIPTION = gql`
  subscription QueueGeneration(
    $prompt: String!
    $steps: Int
    $negativePrompt: String
    $height: Int
    $width: Int
    $seed: Int
    $cfg: Float
    $fluxCfg: Float
    $batchSize: Int
  ) {
    generateImageAndWatch(
      input: {
        numInferenceSteps: $steps
        prompt: $prompt
        negativePrompt: $negativePrompt
        height: $height
        width: $width
        seed: $seed
        cfg: $cfg
        fluxCfg: $fluxCfg
        batchSize: $batchSize
      }
    ) {
      images
      step
      totalSteps
    }
  }
`;

export function generateAndSubscribe(
  prompt: String,
  negativePrompt: String,
  height: Number,
  width: Number,
  seed: Number,
  cfg: Number,
  fluxCfg: Number,
  steps: Number,
  batch: Number,
  callback: Function
) {
  const subscription = client
    .subscribe({
      query: GENERATION_SUBSCRIPTION,
      variables: {
        prompt: prompt,
        negativePrompt: negativePrompt,
        height: height,
        width: width,
        seed: seed,
        cfg: cfg,
        fluxCfg: fluxCfg,
        steps: steps,
        batchSize: batch,
      },
    })
    .subscribe({
      next: ({ data }: { data: any }) => {
        callback(data.generateImageAndWatch);
      },
    });

  return subscription;
}

export function loadModel() {
  return client.query({ query: LOAD_MODEL_QUERY });
}
