from warnings import filterwarnings

filterwarnings("ignore")


import logging  # noqa

from fastapi import FastAPI  # noqa
from fastapi.middleware.cors import CORSMiddleware  # noqa
from fastapi.staticfiles import StaticFiles  # noqa
import strawberry  # noqa
from strawberry.subscriptions import (  # noqa
    GRAPHQL_TRANSPORT_WS_PROTOCOL,
    GRAPHQL_WS_PROTOCOL,
)
from strawberry.fastapi import GraphQLRouter  # noqa
import uvicorn  # noqa

from src import fixes, config  # noqa
from src.schema import Mutations, Queries, Subscriptions  # noqa

fixes.apply_fixes()
config.get_config()  # initialize config

schema = strawberry.Schema(
    query=Queries, mutation=Mutations, subscription=Subscriptions
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = GraphQLRouter(
    schema=schema,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ],
)

app.mount("/images/", StaticFiles(directory="images"), name="images")
app.include_router(router, prefix="/graphql")
app.mount("/", StaticFiles(directory="ui/dist/", html=True), name="html")

if __name__ == "__main__":
    logging.getLogger("uvicorn.asgi").setLevel("DEBUG")
    uvicorn.run(app, host="0.0.0.0", port=8000)
