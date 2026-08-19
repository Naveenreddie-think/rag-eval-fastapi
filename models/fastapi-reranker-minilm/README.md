---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:290
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: fastapi val
      type: fastapi_val
    metrics:
    - type: accuracy
      value: 0.7638888888888888
      name: Accuracy
    - type: accuracy_threshold
      value: 0.6907405257225037
      name: Accuracy Threshold
    - type: f1
      value: 0.5
      name: F1
    - type: f1_threshold
      value: 0.5098562836647034
      name: F1 Threshold
    - type: precision
      value: 0.5625
      name: Precision
    - type: recall
      value: 0.45
      name: Recall
    - type: average_precision
      value: 0.4080411322991797
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ["Do FastAPI's `Depends`-based dependencies work inside WebSocket endpoints the same way they do in regular HTTP path operations?", '## Using `Depends` and others { #using-depends-and-others }\n\nIn WebSocket endpoints you can import from `fastapi` and use:\n\n* `Depends`\n* `Security`\n* `Cookie`\n* `Header`\n* `Path`\n* `Query`\n\nThey work the same way as for other FastAPI endpoints/*path operations*:\n\n```py\nfrom typing import Annotated\n\nfrom fastapi import (\n    Cookie,\n    Depends,\n    FastAPI,\n    Query,\n    WebSocket,\n    WebSocketException,\n    status,\n)\nfrom fastapi.responses import HTMLResponse\n\napp = FastAPI()\n\nhtml = """\n<!DOCTYPE html>\n<html>\n    <head>\n        <title>Chat</title>\n    </head>\n    <body>\n        <h1>WebSocket Chat</h1>\n        <form action="" onsubmit="sendMessage(event)">\n            <label>Item ID: <input type="text" id="itemId" autocomplete="off" value="foo"/></label>\n            <label>Token: <input type="text" id="token" autocomplete="off" value="some-key-token"/></label>\n            <button onclick="connect(event)">Connect</button>\n            <hr>\n            <label>Message: <input type="text" id="messageText" autocomplete="off"/></label>\n            <button>Send</button>\n        </form>\n        <ul id=\'messages\'>\n        </ul>\n        <script>\n        var ws = null;\n            function connect(event) {\n                var itemId = document.getElementById("itemId")\n                var token = document.getElementById("token")\n                ws = new WebSocket("ws://localhost:8000/items/" + itemId.value + "/ws?token=" + token.value);\n                ws.onmessage = function(event) {\n                    var messages = document.getElementById(\'messages\')\n                    var message = document.createElement(\'li\')\n                    var content = document.createTextNode(event.data)\n                    message.appendChild(content)\n                    messages.appendChild(message)\n                };\n                event.preventDefault()\n            }\n            function sendMessage(event) {\n                var input = document.getElementById("messageText")\n                ws.send(input.value)\n                input.value = \'\'\n                event.preventDefault()\n            }\n        </script>\n    </body>\n</html>\n"""\n\n\n@app.get("/")\nasync def get():\n    return HTMLResponse(html)\n\n\nasync def get_cookie_or_token(\n    websocket: WebSocket,\n    session: Annotated[str | None, Cookie()] = None,\n    token: Annotated[str | None, Query()] = None,\n):\n    if session is None and token is None:\n        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)\n    return session or token\n\n\n@app.websocket("/items/{item_id}/ws")\nasync def websocket_endpoint(\n    *,\n    websocket: WebSocket,\n    item_id: str,\n    q: int | None = None,\n    cookie_or_token: Annotated[str, Depends(get_cookie_or_token)],\n):\n    await websocket.accept()\n    while True:\n        data = await websocket.receive_text()\n        await websocket.send_text(\n            f"Session cookie or query token value is: {cookie_or_token}"\n        )\n        if q is not None:\n            await websocket.send_text(f"Query parameter q is: {q}")\n        await websocket.send_text(f"Message text was: {data}, for item ID: {item_id}")\n\n```\n\n/// note\n\nAs this is a WebSocket it doesn\'t really make sense to raise an `HTTPException`, instead we raise a `WebSocketException`.\n\nYou can use a closing code from the [valid codes defined in the specification](https://tools.ietf.org/html/rfc6455#section-7.4.1).\n\n///'],
    ['In the OAuth2-with-JWT tutorial, what happens when `get_current_user` receives an invalid token, and what general FastAPI mechanism is that behavior actually built on?', '## Update the dependencies { #update-the-dependencies }\n\nUpdate `get_current_user` to receive the same token as before, but this time, using JWT tokens.\n\nDecode the received token, verify it, and return the current user.\n\nIf the token is invalid, return an HTTP error right away.\n\n```py\nfrom datetime import datetime, timedelta, timezone\nfrom typing import Annotated\n\nimport jwt\nfrom fastapi import Depends, FastAPI, HTTPException, status\nfrom fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm\nfrom jwt.exceptions import InvalidTokenError\nfrom pwdlib import PasswordHash\nfrom pydantic import BaseModel\n\n# to get a string like this run:\n# openssl rand -hex 32\nSECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"\nALGORITHM = "HS256"\nACCESS_TOKEN_EXPIRE_MINUTES = 30\n\n\nfake_users_db = {\n    "johndoe": {\n        "username": "johndoe",\n        "full_name": "John Doe",\n        "email": "johndoe@example.com",\n        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc",\n        "disabled": False,\n    }\n}\n\n\nclass Token(BaseModel):\n    access_token: str\n    token_type: str\n\n\nclass TokenData(BaseModel):\n    username: str | None = None\n\n\nclass User(BaseModel):\n    username: str\n    email: str | None = None\n    full_name: str | None = None\n    disabled: bool | None = None\n\n\nclass UserInDB(User):\n    hashed_password: str\n\n\npassword_hash = PasswordHash.recommended()\n\nDUMMY_HASH = password_hash.hash("dummypassword")\n\noauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")\n\napp = FastAPI()\n\n\ndef verify_password(plain_password, hashed_password):\n    return password_hash.verify(plain_password, hashed_password)\n\n\ndef get_password_hash(password):\n    return password_hash.hash(password)\n\n\ndef get_user(db, username: str):\n    if username in db:\n        user_dict = db[username]\n        return UserInDB(**user_dict)\n\n\ndef authenticate_user(fake_db, username: str, password: str):\n    user = get_user(fake_db, username)\n    if not user:\n        verify_password(password, DUMMY_HASH)\n        return False\n    if not verify_password(password, user.hashed_password):\n        return False\n    return user\n\n\ndef create_access_token(data: dict, expires_delta: timedelta | None = None):\n    to_encode = data.copy()\n    if expires_delta:\n        expire = datetime.now(timezone.utc) + expires_delta\n    else:\n        expire = datetime.now(timezone.utc) + timedelta(minutes=15)\n    to_encode.update({"exp": expire})\n    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n    return encoded_jwt\n\n\nasync def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):\n    credentials_exception = HTTPException(\n        status_code=status.HTTP_401_UNAUTHORIZED,\n        detail="Could not validate credentials",\n        headers={"WWW-Authenticate": "Bearer"},\n    )\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n        username = payload.get("sub")\n        if username is None:\n            raise credentials_exception\n        token_data = TokenData(username=username)\n    except InvalidTokenError:\n        raise credentials_exception\n    user = get_user(fake_users_db, username=token_data.username)\n    if user is None:\n        raise credentials_exception\n    return user\n\n\nasync def get_current_active_user(\n    current_user: Annotated[User, Depends(get_current_user)],\n):\n    if current_user.disabled:\n        raise HTTPException(status_code=400, detail="Inactive user")\n    return current_user\n\n\n@app.post("/token")\nasync def login_for_access_token(\n    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],\n) -> Token:\n    user = authenticate_user(fake_users_db, form_data.username, form_data.password)\n    if not user:\n        raise HTTPException(\n            status_code=status.HTTP_401_UNAUTHORIZED,\n            detail="Incorrect username or password",\n            headers={"WWW-Authenticate": "Bearer"},\n        )\n    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n    access_token = create_access_token(\n        data={"sub": user.username}, expires_delta=access_token_expires\n    )\n    return Token(access_token=access_token, token_type="bearer")\n\n\n@app.get("/users/me/")\nasync def read_users_me(\n    current_user: Annotated[User, Depends(get_current_active_user)],\n) -> User:\n    return current_user\n\n\n@app.get("/users/me/items/")\nasync def read_own_items(\n    current_user: Annotated[User, Depends(get_current_active_user)],\n):\n    return [{"item_id": "Foo", "owner": current_user.username}]\n\n```'],
    ["The Get Current User tutorial builds `get_current_user` as a dependency with a sub-dependency on `oauth2_scheme`. The Simple OAuth2 tutorial then adds `get_current_active_user` on top. Combining these: what's the actual dependency chain by the end of the Simple OAuth2 tutorial, from the path operation down to the raw token?", '# Get Current User { #get-current-user }\n\nIn the previous chapter the security system (which is based on the dependency injection system) was giving the *path operation function* a `token` as a `str`:\n\n```py\nfrom typing import Annotated\n\nfrom fastapi import Depends, FastAPI\nfrom fastapi.security import OAuth2PasswordBearer\n\napp = FastAPI()\n\noauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")\n\n\n@app.get("/items/")\nasync def read_items(token: Annotated[str, Depends(oauth2_scheme)]):\n    return {"token": token}\n\n```\n\nBut that is still not that useful.\n\nLet\'s make it give us the current user.'],
    ["In FastAPI's OAuth2 scopes example, what type does the `security_scopes` parameter have inside the `get_current_user` dependency, and what property does it expose listing all required scopes?", '## Use the `scopes` { #use-the-scopes }\n\nThe parameter `security_scopes` will be of type `SecurityScopes`.\n\nIt will have a property `scopes` with a list containing all the scopes required by itself and all the dependencies that use this as a sub-dependency. That means, all the "dependants"... this might sound confusing, it is explained again later below.\n\nThe `security_scopes` object (of class `SecurityScopes`) also provides a `scope_str` attribute with a single string, containing those scopes separated by spaces (we are going to use it).\n\nWe create an `HTTPException` that we can reuse (`raise`) later at several points.\n\nIn this exception, we include the scopes required (if any) as a string separated by spaces (using `scope_str`). We put that string containing the scopes in the `WWW-Authenticate` header (this is part of the spec).\n\n```py\nfrom datetime import datetime, timedelta, timezone\nfrom typing import Annotated\n\nimport jwt\nfrom fastapi import Depends, FastAPI, HTTPException, Security, status\nfrom fastapi.security import (\n    OAuth2PasswordBearer,\n    OAuth2PasswordRequestForm,\n    SecurityScopes,\n)\nfrom jwt.exceptions import InvalidTokenError\nfrom pwdlib import PasswordHash\nfrom pydantic import BaseModel, ValidationError\n\n# to get a string like this run:\n# openssl rand -hex 32\nSECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"\nALGORITHM = "HS256"\nACCESS_TOKEN_EXPIRE_MINUTES = 30\n\n\nfake_users_db = {\n    "johndoe": {\n        "username": "johndoe",\n        "full_name": "John Doe",\n        "email": "johndoe@example.com",\n        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc",\n        "disabled": False,\n    },\n    "alice": {\n        "username": "alice",\n        "full_name": "Alice Chains",\n        "email": "alicechains@example.com",\n        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$g2/AV1zwopqUntPKJavBFw$BwpRGDCyUHLvHICnwijyX8ROGoiUPwNKZ7915MeYfCE",\n        "disabled": True,\n    },\n}\n\n\nclass Token(BaseModel):\n    access_token: str\n    token_type: str\n\n\nclass TokenData(BaseModel):\n    username: str | None = None\n    scopes: list[str] = []\n\n\nclass User(BaseModel):\n    username: str\n    email: str | None = None\n    full_name: str | None = None\n    disabled: bool | None = None\n\n\nclass UserInDB(User):\n    hashed_password: str\n\n\npassword_hash = PasswordHash.recommended()\n\nDUMMY_HASH = password_hash.hash("dummypassword")\n\noauth2_scheme = OAuth2PasswordBearer(\n    tokenUrl="token",\n    scopes={"me": "Read information about the current user.", "items": "Read items."},\n)\n\napp = FastAPI()\n\n\ndef verify_password(plain_password, hashed_password):\n    return password_hash.verify(plain_password, hashed_password)\n\n\ndef get_password_hash(password):\n    return password_hash.hash(password)\n\n\ndef get_user(db, username: str):\n    if username in db:\n        user_dict = db[username]\n        return UserInDB(**user_dict)\n\n\ndef authenticate_user(fake_db, username: str, password: str):\n    user = get_user(fake_db, username)\n    if not user:\n        verify_password(password, DUMMY_HASH)\n        return False\n    if not verify_password(password, user.hashed_password):\n        return False\n    return user\n\n\ndef create_access_token(data: dict, expires_delta: timedelta | None = None):\n    to_encode = data.copy()\n    if expires_delta:\n        expire = datetime.now(timezone.utc) + expires_delta\n    else:\n        expire = datetime.now(timezone.utc) + timedelta(minutes=15)\n    to_encode.update({"exp": expire})\n    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n    return encoded_jwt\n\n\nasync def get_current_user(\n    security_scopes: SecurityScopes, token: Annotated[str, Depends(oauth2_scheme)]\n):\n    if security_scopes.scopes:\n        authenticate_value = f\'Bearer scope="{security_scopes.scope_str}"\'\n    else:\n        authenticate_value = "Bearer"\n    credentials_exception = HTTPException(\n        status_code=status.HTTP_401_UNAUTHORIZED,\n        detail="Could not validate credentials",\n        headers={"WWW-Authenticate": authenticate_value},\n    )\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n        username = payload.get("sub")\n        if username is None:\n            raise credentials_exception\n        scope: str = payload.get("scope", "")\n        token_scopes = scope.split(" ")\n        token_data = TokenData(scopes=token_scopes, username=username)\n    except (InvalidTokenError, ValidationError):\n        raise credentials_exception\n    user = get_user(fake_users_db, username=token_data.username)\n    if user is None:\n        raise credentials_exception\n    for scope in security_scopes.scopes:\n        if scope not in token_data.scopes:\n            raise HTTPException(\n                status_code=status.HTTP_401_UNAUTHORIZED,\n                detail="Not enough permissions",\n                headers={"WWW-Authenticate": authenticate_value},\n            )\n    return user\n\n\nasync def get_current_active_user(\n    current_user: Annotated[User, Security(get_current_user, scopes=["me"])],\n):\n    if current_user.disabled:\n        raise HTTPException(status_code=400, detail="Inactive user")\n    return current_user\n\n\n@app.post("/token")\nasync def login_for_access_token(\n    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],\n) -> Token:\n    user = authenticate_user(fake_users_db, form_data.username, form_data.password)\n    if not user:\n        raise HTTPException(status_code=400, detail="Incorrect username or password")\n    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n    access_token = create_access_token(\n        data={"sub": user.username, "scope": " ".join(form_data.scopes)},\n        expires_delta=access_token_expires,\n    )\n    return Token(access_token=access_token, token_type="bearer")\n\n\n@app.get("/users/me/")\nasync def read_users_me(\n    current_user: Annotated[User, Depends(get_current_active_user)],\n) -> User:\n    return current_user\n\n\n@app.get("/users/me/items/")\nasync def read_own_items(\n    current_user: Annotated[User, Security(get_current_active_user, scopes=["items"])],\n):\n    return [{"item_id": "Foo", "owner": current_user.username}]\n\n\n@app.get("/status/")\nasync def read_system_status(current_user: Annotated[User, Depends(get_current_user)]):\n    return {"status": "ok"}\n\n```'],
    ['How do you set a maximum length of 50 characters on an optional query parameter `q`?', '## Optional parameters { #optional-parameters }\n\nThe same way, you can declare optional query parameters, by setting their default to `None`:\n\n```py\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get("/items/{item_id}")\nasync def read_item(item_id: str, q: str | None = None):\n    if q:\n        return {"item_id": item_id, "q": q}\n    return {"item_id": item_id}\n\n```\n\nIn this case, the function parameter `q` will be optional, and will be `None` by default.\n\n/// tip\n\nAlso notice that **FastAPI** is smart enough to notice that the path parameter `item_id` is a path parameter and `q` is not, so, it\'s a query parameter.\n\n///'],
]
scores = model.predict(pairs)
print(scores)
# [ 1.3809  0.3292  0.4062 -0.1342 -1.626 ]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "Do FastAPI's `Depends`-based dependencies work inside WebSocket endpoints the same way they do in regular HTTP path operations?",
    [
        '## Using `Depends` and others { #using-depends-and-others }\n\nIn WebSocket endpoints you can import from `fastapi` and use:\n\n* `Depends`\n* `Security`\n* `Cookie`\n* `Header`\n* `Path`\n* `Query`\n\nThey work the same way as for other FastAPI endpoints/*path operations*:\n\n```py\nfrom typing import Annotated\n\nfrom fastapi import (\n    Cookie,\n    Depends,\n    FastAPI,\n    Query,\n    WebSocket,\n    WebSocketException,\n    status,\n)\nfrom fastapi.responses import HTMLResponse\n\napp = FastAPI()\n\nhtml = """\n<!DOCTYPE html>\n<html>\n    <head>\n        <title>Chat</title>\n    </head>\n    <body>\n        <h1>WebSocket Chat</h1>\n        <form action="" onsubmit="sendMessage(event)">\n            <label>Item ID: <input type="text" id="itemId" autocomplete="off" value="foo"/></label>\n            <label>Token: <input type="text" id="token" autocomplete="off" value="some-key-token"/></label>\n            <button onclick="connect(event)">Connect</button>\n            <hr>\n            <label>Message: <input type="text" id="messageText" autocomplete="off"/></label>\n            <button>Send</button>\n        </form>\n        <ul id=\'messages\'>\n        </ul>\n        <script>\n        var ws = null;\n            function connect(event) {\n                var itemId = document.getElementById("itemId")\n                var token = document.getElementById("token")\n                ws = new WebSocket("ws://localhost:8000/items/" + itemId.value + "/ws?token=" + token.value);\n                ws.onmessage = function(event) {\n                    var messages = document.getElementById(\'messages\')\n                    var message = document.createElement(\'li\')\n                    var content = document.createTextNode(event.data)\n                    message.appendChild(content)\n                    messages.appendChild(message)\n                };\n                event.preventDefault()\n            }\n            function sendMessage(event) {\n                var input = document.getElementById("messageText")\n                ws.send(input.value)\n                input.value = \'\'\n                event.preventDefault()\n            }\n        </script>\n    </body>\n</html>\n"""\n\n\n@app.get("/")\nasync def get():\n    return HTMLResponse(html)\n\n\nasync def get_cookie_or_token(\n    websocket: WebSocket,\n    session: Annotated[str | None, Cookie()] = None,\n    token: Annotated[str | None, Query()] = None,\n):\n    if session is None and token is None:\n        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)\n    return session or token\n\n\n@app.websocket("/items/{item_id}/ws")\nasync def websocket_endpoint(\n    *,\n    websocket: WebSocket,\n    item_id: str,\n    q: int | None = None,\n    cookie_or_token: Annotated[str, Depends(get_cookie_or_token)],\n):\n    await websocket.accept()\n    while True:\n        data = await websocket.receive_text()\n        await websocket.send_text(\n            f"Session cookie or query token value is: {cookie_or_token}"\n        )\n        if q is not None:\n            await websocket.send_text(f"Query parameter q is: {q}")\n        await websocket.send_text(f"Message text was: {data}, for item ID: {item_id}")\n\n```\n\n/// note\n\nAs this is a WebSocket it doesn\'t really make sense to raise an `HTTPException`, instead we raise a `WebSocketException`.\n\nYou can use a closing code from the [valid codes defined in the specification](https://tools.ietf.org/html/rfc6455#section-7.4.1).\n\n///',
        '## Update the dependencies { #update-the-dependencies }\n\nUpdate `get_current_user` to receive the same token as before, but this time, using JWT tokens.\n\nDecode the received token, verify it, and return the current user.\n\nIf the token is invalid, return an HTTP error right away.\n\n```py\nfrom datetime import datetime, timedelta, timezone\nfrom typing import Annotated\n\nimport jwt\nfrom fastapi import Depends, FastAPI, HTTPException, status\nfrom fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm\nfrom jwt.exceptions import InvalidTokenError\nfrom pwdlib import PasswordHash\nfrom pydantic import BaseModel\n\n# to get a string like this run:\n# openssl rand -hex 32\nSECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"\nALGORITHM = "HS256"\nACCESS_TOKEN_EXPIRE_MINUTES = 30\n\n\nfake_users_db = {\n    "johndoe": {\n        "username": "johndoe",\n        "full_name": "John Doe",\n        "email": "johndoe@example.com",\n        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc",\n        "disabled": False,\n    }\n}\n\n\nclass Token(BaseModel):\n    access_token: str\n    token_type: str\n\n\nclass TokenData(BaseModel):\n    username: str | None = None\n\n\nclass User(BaseModel):\n    username: str\n    email: str | None = None\n    full_name: str | None = None\n    disabled: bool | None = None\n\n\nclass UserInDB(User):\n    hashed_password: str\n\n\npassword_hash = PasswordHash.recommended()\n\nDUMMY_HASH = password_hash.hash("dummypassword")\n\noauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")\n\napp = FastAPI()\n\n\ndef verify_password(plain_password, hashed_password):\n    return password_hash.verify(plain_password, hashed_password)\n\n\ndef get_password_hash(password):\n    return password_hash.hash(password)\n\n\ndef get_user(db, username: str):\n    if username in db:\n        user_dict = db[username]\n        return UserInDB(**user_dict)\n\n\ndef authenticate_user(fake_db, username: str, password: str):\n    user = get_user(fake_db, username)\n    if not user:\n        verify_password(password, DUMMY_HASH)\n        return False\n    if not verify_password(password, user.hashed_password):\n        return False\n    return user\n\n\ndef create_access_token(data: dict, expires_delta: timedelta | None = None):\n    to_encode = data.copy()\n    if expires_delta:\n        expire = datetime.now(timezone.utc) + expires_delta\n    else:\n        expire = datetime.now(timezone.utc) + timedelta(minutes=15)\n    to_encode.update({"exp": expire})\n    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n    return encoded_jwt\n\n\nasync def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):\n    credentials_exception = HTTPException(\n        status_code=status.HTTP_401_UNAUTHORIZED,\n        detail="Could not validate credentials",\n        headers={"WWW-Authenticate": "Bearer"},\n    )\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n        username = payload.get("sub")\n        if username is None:\n            raise credentials_exception\n        token_data = TokenData(username=username)\n    except InvalidTokenError:\n        raise credentials_exception\n    user = get_user(fake_users_db, username=token_data.username)\n    if user is None:\n        raise credentials_exception\n    return user\n\n\nasync def get_current_active_user(\n    current_user: Annotated[User, Depends(get_current_user)],\n):\n    if current_user.disabled:\n        raise HTTPException(status_code=400, detail="Inactive user")\n    return current_user\n\n\n@app.post("/token")\nasync def login_for_access_token(\n    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],\n) -> Token:\n    user = authenticate_user(fake_users_db, form_data.username, form_data.password)\n    if not user:\n        raise HTTPException(\n            status_code=status.HTTP_401_UNAUTHORIZED,\n            detail="Incorrect username or password",\n            headers={"WWW-Authenticate": "Bearer"},\n        )\n    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n    access_token = create_access_token(\n        data={"sub": user.username}, expires_delta=access_token_expires\n    )\n    return Token(access_token=access_token, token_type="bearer")\n\n\n@app.get("/users/me/")\nasync def read_users_me(\n    current_user: Annotated[User, Depends(get_current_active_user)],\n) -> User:\n    return current_user\n\n\n@app.get("/users/me/items/")\nasync def read_own_items(\n    current_user: Annotated[User, Depends(get_current_active_user)],\n):\n    return [{"item_id": "Foo", "owner": current_user.username}]\n\n```',
        '# Get Current User { #get-current-user }\n\nIn the previous chapter the security system (which is based on the dependency injection system) was giving the *path operation function* a `token` as a `str`:\n\n```py\nfrom typing import Annotated\n\nfrom fastapi import Depends, FastAPI\nfrom fastapi.security import OAuth2PasswordBearer\n\napp = FastAPI()\n\noauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")\n\n\n@app.get("/items/")\nasync def read_items(token: Annotated[str, Depends(oauth2_scheme)]):\n    return {"token": token}\n\n```\n\nBut that is still not that useful.\n\nLet\'s make it give us the current user.',
        '## Use the `scopes` { #use-the-scopes }\n\nThe parameter `security_scopes` will be of type `SecurityScopes`.\n\nIt will have a property `scopes` with a list containing all the scopes required by itself and all the dependencies that use this as a sub-dependency. That means, all the "dependants"... this might sound confusing, it is explained again later below.\n\nThe `security_scopes` object (of class `SecurityScopes`) also provides a `scope_str` attribute with a single string, containing those scopes separated by spaces (we are going to use it).\n\nWe create an `HTTPException` that we can reuse (`raise`) later at several points.\n\nIn this exception, we include the scopes required (if any) as a string separated by spaces (using `scope_str`). We put that string containing the scopes in the `WWW-Authenticate` header (this is part of the spec).\n\n```py\nfrom datetime import datetime, timedelta, timezone\nfrom typing import Annotated\n\nimport jwt\nfrom fastapi import Depends, FastAPI, HTTPException, Security, status\nfrom fastapi.security import (\n    OAuth2PasswordBearer,\n    OAuth2PasswordRequestForm,\n    SecurityScopes,\n)\nfrom jwt.exceptions import InvalidTokenError\nfrom pwdlib import PasswordHash\nfrom pydantic import BaseModel, ValidationError\n\n# to get a string like this run:\n# openssl rand -hex 32\nSECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"\nALGORITHM = "HS256"\nACCESS_TOKEN_EXPIRE_MINUTES = 30\n\n\nfake_users_db = {\n    "johndoe": {\n        "username": "johndoe",\n        "full_name": "John Doe",\n        "email": "johndoe@example.com",\n        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc",\n        "disabled": False,\n    },\n    "alice": {\n        "username": "alice",\n        "full_name": "Alice Chains",\n        "email": "alicechains@example.com",\n        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$g2/AV1zwopqUntPKJavBFw$BwpRGDCyUHLvHICnwijyX8ROGoiUPwNKZ7915MeYfCE",\n        "disabled": True,\n    },\n}\n\n\nclass Token(BaseModel):\n    access_token: str\n    token_type: str\n\n\nclass TokenData(BaseModel):\n    username: str | None = None\n    scopes: list[str] = []\n\n\nclass User(BaseModel):\n    username: str\n    email: str | None = None\n    full_name: str | None = None\n    disabled: bool | None = None\n\n\nclass UserInDB(User):\n    hashed_password: str\n\n\npassword_hash = PasswordHash.recommended()\n\nDUMMY_HASH = password_hash.hash("dummypassword")\n\noauth2_scheme = OAuth2PasswordBearer(\n    tokenUrl="token",\n    scopes={"me": "Read information about the current user.", "items": "Read items."},\n)\n\napp = FastAPI()\n\n\ndef verify_password(plain_password, hashed_password):\n    return password_hash.verify(plain_password, hashed_password)\n\n\ndef get_password_hash(password):\n    return password_hash.hash(password)\n\n\ndef get_user(db, username: str):\n    if username in db:\n        user_dict = db[username]\n        return UserInDB(**user_dict)\n\n\ndef authenticate_user(fake_db, username: str, password: str):\n    user = get_user(fake_db, username)\n    if not user:\n        verify_password(password, DUMMY_HASH)\n        return False\n    if not verify_password(password, user.hashed_password):\n        return False\n    return user\n\n\ndef create_access_token(data: dict, expires_delta: timedelta | None = None):\n    to_encode = data.copy()\n    if expires_delta:\n        expire = datetime.now(timezone.utc) + expires_delta\n    else:\n        expire = datetime.now(timezone.utc) + timedelta(minutes=15)\n    to_encode.update({"exp": expire})\n    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n    return encoded_jwt\n\n\nasync def get_current_user(\n    security_scopes: SecurityScopes, token: Annotated[str, Depends(oauth2_scheme)]\n):\n    if security_scopes.scopes:\n        authenticate_value = f\'Bearer scope="{security_scopes.scope_str}"\'\n    else:\n        authenticate_value = "Bearer"\n    credentials_exception = HTTPException(\n        status_code=status.HTTP_401_UNAUTHORIZED,\n        detail="Could not validate credentials",\n        headers={"WWW-Authenticate": authenticate_value},\n    )\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n        username = payload.get("sub")\n        if username is None:\n            raise credentials_exception\n        scope: str = payload.get("scope", "")\n        token_scopes = scope.split(" ")\n        token_data = TokenData(scopes=token_scopes, username=username)\n    except (InvalidTokenError, ValidationError):\n        raise credentials_exception\n    user = get_user(fake_users_db, username=token_data.username)\n    if user is None:\n        raise credentials_exception\n    for scope in security_scopes.scopes:\n        if scope not in token_data.scopes:\n            raise HTTPException(\n                status_code=status.HTTP_401_UNAUTHORIZED,\n                detail="Not enough permissions",\n                headers={"WWW-Authenticate": authenticate_value},\n            )\n    return user\n\n\nasync def get_current_active_user(\n    current_user: Annotated[User, Security(get_current_user, scopes=["me"])],\n):\n    if current_user.disabled:\n        raise HTTPException(status_code=400, detail="Inactive user")\n    return current_user\n\n\n@app.post("/token")\nasync def login_for_access_token(\n    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],\n) -> Token:\n    user = authenticate_user(fake_users_db, form_data.username, form_data.password)\n    if not user:\n        raise HTTPException(status_code=400, detail="Incorrect username or password")\n    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n    access_token = create_access_token(\n        data={"sub": user.username, "scope": " ".join(form_data.scopes)},\n        expires_delta=access_token_expires,\n    )\n    return Token(access_token=access_token, token_type="bearer")\n\n\n@app.get("/users/me/")\nasync def read_users_me(\n    current_user: Annotated[User, Depends(get_current_active_user)],\n) -> User:\n    return current_user\n\n\n@app.get("/users/me/items/")\nasync def read_own_items(\n    current_user: Annotated[User, Security(get_current_active_user, scopes=["items"])],\n):\n    return [{"item_id": "Foo", "owner": current_user.username}]\n\n\n@app.get("/status/")\nasync def read_system_status(current_user: Annotated[User, Depends(get_current_user)]):\n    return {"status": "ok"}\n\n```',
        '## Optional parameters { #optional-parameters }\n\nThe same way, you can declare optional query parameters, by setting their default to `None`:\n\n```py\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get("/items/{item_id}")\nasync def read_item(item_id: str, q: str | None = None):\n    if q:\n        return {"item_id": item_id, "q": q}\n    return {"item_id": item_id}\n\n```\n\nIn this case, the function parameter `q` will be optional, and will be `None` by default.\n\n/// tip\n\nAlso notice that **FastAPI** is smart enough to notice that the path parameter `item_id` is a path parameter and `q` is not, so, it\'s a query parameter.\n\n///',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `fastapi_val`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value     |
|:----------------------|:----------|
| accuracy              | 0.7639    |
| accuracy_threshold    | 0.6907    |
| f1                    | 0.5       |
| f1_threshold          | 0.5099    |
| precision             | 0.5625    |
| recall                | 0.45      |
| **average_precision** | **0.408** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 290 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:---------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                            | string                                                                               | float                                                          |
  | modality | text                                                                              | text                                                                                 |                                                                |
  | details  | <ul><li>min: 19 tokens</li><li>mean: 44.8 tokens</li><li>max: 99 tokens</li></ul> | <ul><li>min: 35 tokens</li><li>mean: 282.09 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.25</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Do FastAPI's `Depends`-based dependencies work inside WebSocket endpoints the same way they do in regular HTTP path operations?</code>                                                                                                                                                                                                       | <code>## Using `Depends` and others { #using-depends-and-others }<br><br>In WebSocket endpoints you can import from `fastapi` and use:<br><br>* `Depends`<br>* `Security`<br>* `Cookie`<br>* `Header`<br>* `Path`<br>* `Query`<br><br>They work the same way as for other FastAPI endpoints/*path operations*:<br><br>```py<br>from typing import Annotated<br><br>from fastapi import (<br>    Cookie,<br>    Depends,<br>    FastAPI,<br>    Query,<br>    WebSocket,<br>    WebSocketException,<br>    status,<br>)<br>from fastapi.responses import HTMLResponse<br><br>app = FastAPI()<br><br>html = """<br><!DOCTYPE html><br><html><br>    <head><br>        <title>Chat</title><br>    </head><br>    <body><br>        <h1>WebSocket Chat</h1><br>        <form action="" onsubmit="sendMessage(event)"><br>            <label>Item ID: <input type="text" id="itemId" autocomplete="off" value="foo"/></label><br>            <label>Token: <input type="text" id="token" autocomplete="off" value="some-key-token"/></label><br>            <button onclick="connect(event)">Connect</button><br>            <hr><br>            <label>Message: <input type="tex...</code> | <code>1.0</code> |
  | <code>In the OAuth2-with-JWT tutorial, what happens when `get_current_user` receives an invalid token, and what general FastAPI mechanism is that behavior actually built on?</code>                                                                                                                                                               | <code>## Update the dependencies { #update-the-dependencies }<br><br>Update `get_current_user` to receive the same token as before, but this time, using JWT tokens.<br><br>Decode the received token, verify it, and return the current user.<br><br>If the token is invalid, return an HTTP error right away.<br><br>```py<br>from datetime import datetime, timedelta, timezone<br>from typing import Annotated<br><br>import jwt<br>from fastapi import Depends, FastAPI, HTTPException, status<br>from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm<br>from jwt.exceptions import InvalidTokenError<br>from pwdlib import PasswordHash<br>from pydantic import BaseModel<br><br># to get a string like this run:<br># openssl rand -hex 32<br>SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"<br>ALGORITHM = "HS256"<br>ACCESS_TOKEN_EXPIRE_MINUTES = 30<br><br><br>fake_users_db = {<br>    "johndoe": {<br>        "username": "johndoe",<br>        "full_name": "John Doe",<br>        "email": "johndoe@example.com",<br>        "hashed_password": "$argon2id$v=19$m=...</code>                                  | <code>1.0</code> |
  | <code>The Get Current User tutorial builds `get_current_user` as a dependency with a sub-dependency on `oauth2_scheme`. The Simple OAuth2 tutorial then adds `get_current_active_user` on top. Combining these: what's the actual dependency chain by the end of the Simple OAuth2 tutorial, from the path operation down to the raw token?</code> | <code># Get Current User { #get-current-user }<br><br>In the previous chapter the security system (which is based on the dependency injection system) was giving the *path operation function* a `token` as a `str`:<br><br>```py<br>from typing import Annotated<br><br>from fastapi import Depends, FastAPI<br>from fastapi.security import OAuth2PasswordBearer<br><br>app = FastAPI()<br><br>oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")<br><br><br>@app.get("/items/")<br>async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):<br>    return {"token": token}<br><br>```<br><br>But that is still not that useful.<br><br>Let's make it give us the current user.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | fastapi_val_average_precision |
|:------:|:----:|:-----------------------------:|
| 0.5405 | 20   | 0.3535                        |
| 1.0    | 37   | 0.3805                        |
| 1.0811 | 40   | 0.3824                        |
| 1.6216 | 60   | 0.3991                        |
| 2.0    | 74   | 0.4054                        |
| 2.1622 | 80   | 0.4018                        |
| 2.7027 | 100  | 0.4071                        |
| 3.0    | 111  | 0.4080                        |


### Training Time
- **Training**: 13.4 seconds
- **Evaluation**: 1.2 seconds
- **Total**: 14.7 seconds

### Framework Versions
- Python: 3.12.10
- Sentence Transformers: 5.6.0
- Transformers: 5.5.0
- PyTorch: 2.12.0.dev20260408+cu128
- Accelerate: 1.14.0
- Datasets: 4.3.0
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->