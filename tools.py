from fastapi import Request

async def parse_starlette_request(request: Request):
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")
    headers = dict(request.headers)
    client_host = request.client.host

    return {
        "client_host": client_host,
        "headers": headers,
        "body": body_str
    }