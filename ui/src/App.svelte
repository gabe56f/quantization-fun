<script lang="ts">
  import "./app.css";
  import { generateAndSubscribe, loadModel } from "./server";
  import type { Generation } from "./server";

  let prompt = "a cat and a dog having a cartoon fight";
  let negativePrompt = "";
  let seed = -1;
  let cfg = 1.0;
  let fluxCfg = 3.5;
  let width = 1024;
  let height = 1024;
  let steps = 28;
  let batchSize = 1;

  let image = "";

  const gen = () => {
    generateAndSubscribe(
      prompt,
      negativePrompt,
      height,
      width,
      seed,
      cfg,
      fluxCfg,
      steps,
      batchSize,
      (data: Generation) => {
        console.log(data);
        image = data.images[0];
      }
    );
  };

  const load = () => {
    loadModel();
  };
</script>

<main>
  <div class="size-96 relative">
    <div class="w-full h-full">
      <!-- svelte-ignore a11y-img-redundant-alt -->
      <img src={image} class="object-fill h-96 w-96" alt="generated picture" />
    </div>
  </div>
  <input type="text" bind:value={prompt} placeholder="prompt" class="input" />
  <input
    type="text"
    bind:value={negativePrompt}
    placeholder="negative prompt"
    class="input"
  />
  <input type="number" bind:value={seed} placeholder="seed" class="input" />
  <input type="number" bind:value={cfg} placeholder="cfg" class="input" />
  <input
    type="number"
    bind:value={fluxCfg}
    placeholder="fluxCfg"
    class="input"
  />
  <input type="number" bind:value={width} placeholder="width" class="input" />
  <input type="number" bind:value={height} placeholder="height" class="input" />
  <input type="number" bind:value={steps} placeholder="steps" class="input" />
  <input
    type="number"
    bind:value={batchSize}
    placeholder="batchSize"
    class="input"
  />

  <div />
  <button on:click={gen}>generate</button>
  <button on:click={load}>load model</button>
</main>
