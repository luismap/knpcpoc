import os
import json
import cmlapi


CDSW_DOMAIN = os.environ["CDSW_DOMAIN"]
CDSW_APIV2_KEY = os.environ["CDSW_APIV2_KEY"]
CDSW_PROJECT_ID = os.environ["CDSW_PROJECT_ID"]
HEADERS = {"Content-Type": "application/json"}
MODEL_API_URL = f"https://modelservice.{CDSW_DOMAIN}/model"
WORKSPACE_DOMAIN = f"https://{CDSW_DOMAIN}"
CML_CLIENT = cmlapi.default_client(WORKSPACE_DOMAIN, CDSW_APIV2_KEY)


def get_model_access_key(model_search_string):
    models = CML_CLIENT.list_models(CDSW_PROJECT_ID).models
    model = list(filter(lambda m: m.name == model_search_string, models))
    MODEL_ACCESS_KEY = model.access_key

    return MODEL_ACCESS_KEY
