import requests

def graphql_requester(query, variables=None, headers=None):
    """
    Sends a POST request to a GraphQL endpoint.

    Args:
        query (str): GraphQL query or mutation.
        variables (dict, optional): Variables for the query. Defaults to None.
        headers (dict, optional): HTTP headers. Defaults to Content-Type: application/json.

    Returns:
        dict: JSON response from the GraphQL server.
    """
    url = "https://coral-app-fmgao.ondigitalocean.app/graphql"

    if headers is None:
        headers = {
            "Content-Type": "application/json"
        }

    payload = {
        "query": query,
        "variables": variables or {}
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get("data")
    else:
        raise Exception(f"GraphQL request failed with status {response.status_code}: {response.text}")
