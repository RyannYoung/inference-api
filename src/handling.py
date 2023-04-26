from flask import Response

from lib import serve_pil_image


def api_error(err: str, status_code: int = 400) -> Response:
    """Wrapper function to log an error to an api endpoint
    as a JSON response

    Args:
        err (str): The error message
        status_code (int, optional): The optional HTTP response status code
        Defaults to 400.
    """
    return Response(
        response=f'{{"error": {err}}}', status=status_code, mimetype="application/json"
    )


def json_response(data, status_code: int = 400) -> Response:
    return Response(response=data, status=status_code, mimetype="application/json")


def format_response(data, format) -> Response:
    """Serves a response based on the format.
    I.e., if it is a json then return as application/json

    Args:
        data (any): The data to respond with
        format (any): The format to respond as
    """
    match format:
        case "img" | "image" | "pic":
            return serve_pil_image(data)
        case "json" | "dict" | "raw" | "text":
            return data  # Python should parse this as json
        case _:
            return api_error(
                "A response was created but encountered an issue, perhaps your 'format' query is invalid?"
            )
