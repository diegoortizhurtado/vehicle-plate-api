# main.py
from typing import Optional
from datetime import datetime
import secrets
import hashlib

from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
from sqlmodel import SQLModel, Field, create_engine, Session, select
from passlib.context import CryptContext

# ---------- Config ----------
DATABASE_URL = "sqlite:///./railey.db"
engine = create_engine(DATABASE_URL, echo=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="Railey Auth API")

# ---------- 🔐 CORS middleware ----------
origins = [
    "http://localhost:5173",  # React local (vite)
    "https://vehicle-plate-app-production.up.railway.app",  # tu frontend desplegado
    "https://vehicle-plate-api-production.up.railway.app",  # tu API pública
]

# Si quieres permitir cualquier origen durante desarrollo (más simple):
# origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],            # permite todos los métodos HTTP
    allow_headers=["*"],            # permite todos los headers
)

# ---------- DB models ----------
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, nullable=False, unique=True)
    hashed_password: str
    api_key_hash: Optional[str] = None  # SHA256 hash of API key
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ---------- Pydantic schemas ----------
class UserCreate(BaseModel):
    username: constr(min_length=3, max_length=50)
    password: constr(min_length=6)

class UserRead(BaseModel):
    id: int
    username: str
    is_active: bool
    created_at: datetime

class APIKeyResponse(BaseModel):
    api_key: str  # shown one time

class LoginRequest(BaseModel):
    username: str
    password: str

class RegenerateKeyRequest(BaseModel):
    username: str
    password: str

# ---------- Utils ----------
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def generate_api_key() -> str:
    # token_urlsafe is URL safe and decently random
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    # store SHA256 hex digest of the key
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

# ---------- DB helpers ----------
def get_user_by_username(session: Session, username: str) -> Optional[User]:
    statement = select(User).where(User.username == username)
    return session.exec(statement).first()

def get_user_by_api_key_hash(session: Session, api_key_hash: str) -> Optional[User]:
    statement = select(User).where(User.api_key_hash == api_key_hash)
    return session.exec(statement).first()

# ---------- Auth dependency ----------
def get_current_user(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> User:
    """
    Permite Authorization: Bearer <api_key> OR X-API-Key: <api_key>.
    Devuelve el objeto User o lanza 401.
    """
    # extrae token del header Authorization si viene con Bearer
    token = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization:
        # formato esperado: "Bearer <token>"
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing API key (X-API-Key or Authorization: Bearer <key>)",
                            headers={"WWW-Authenticate": "Bearer"})

    token_hash = hash_api_key(token)
    with Session(engine) as session:
        user = get_user_by_api_key_hash(session, token_hash)
        if not user or not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Invalid or inactive API key",
                                headers={"WWW-Authenticate": "Bearer"})
        return user

# ---------- Endpoints ----------
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/users", response_model=APIKeyResponse, status_code=201)
def create_user(payload: UserCreate):
    """
    Crea un usuario. Devuelve la API key sólo una vez (mostrarla al usuario).
    """
    with Session(engine) as session:
        existing = get_user_by_username(session, payload.username)
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")

        hashed_pw = get_password_hash(payload.password)
        api_key_plain = generate_api_key()
        api_key_hashed = hash_api_key(api_key_plain)

        user = User(
            username=payload.username,
            hashed_password=hashed_pw,
            api_key_hash=api_key_hashed,
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        # Se devuelve la api_key en texto claro **una sola vez**
        return APIKeyResponse(api_key=api_key_plain)

@app.post("/login", response_model=APIKeyResponse)
def login(req: LoginRequest):
    """
    Login por username+password -> regenera y devuelve una nueva API key.
    (Esto evita almacenar la API key en texto y permite al usuario recuperar acceso.)
    """
    with Session(engine) as session:
        user = get_user_by_username(session, req.username)
        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        # Regenerar clave y devolverla
        new_key = generate_api_key()
        user.api_key_hash = hash_api_key(new_key)
        session.add(user)
        session.commit()
        return APIKeyResponse(api_key=new_key)

@app.post("/users/{user_id}/regen-key", response_model=APIKeyResponse)
def regenerate_key(user_id: int, req: RegenerateKeyRequest):
    """
    Endpoint alternativo: para regenerar API key de un user (requiere usuario+password).
    Útil si el usuario pide renovar la clave de forma manual.
    """
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.username != req.username or not verify_password(req.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        new_key = generate_api_key()
        user.api_key_hash = hash_api_key(new_key)
        session.add(user)
        session.commit()
        return APIKeyResponse(api_key=new_key)

@app.get("/me", response_model=UserRead)
def read_me(current_user: User = Depends(get_current_user)):
    """Devuelve información del usuario autenticado (no devuelve la API key)."""
    return UserRead(
        id=current_user.id,
        username=current_user.username,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

@app.get("/protected")
def protected_endpoint(current_user: User = Depends(get_current_user)):
    """Ejemplo de endpoint protegido por API key."""
    return {"message": f"Hola {current_user.username}, autenticación exitosa."}

# Endpoint admin simple: list users (en producción proteger con roles)
@app.get("/admin/users")
def list_users(current_user: User = Depends(get_current_user)):
    """
    Ejemplo: lista usuarios. En producción deberías añadir verificación de rol/admin.
    Aquí sólo se permite si el usuario autenticado existe (sólo demo).
    """
    with Session(engine) as session:
        users = session.exec(select(User)).all()
        return [{"id": u.id, "username": u.username, "is_active": u.is_active, "created_at": u.created_at.isoformat()} for u in users]