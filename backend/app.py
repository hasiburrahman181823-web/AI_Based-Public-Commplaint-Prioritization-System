"""Transactional FastAPI service. Run from the repository root with --factory.

Each synchronous route runs in FastAPI's worker pool. MLService's reentrant
lock protects the per-worker analyzer/index; a database transaction lock
serializes write/refresh/inference across workers. No user-supplied group IDs,
scores, or ML outputs are accepted on the citizen submission endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from datetime import datetime
import logging
from pathlib import Path
from uuid import UUID

from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import func, inspect, or_, select, text as sql_text
from sqlalchemy.exc import SQLAlchemyError

from backend.config import ROOT, Settings
from backend.db import make_engine, make_session_factory
from backend.ml_service import MLService
from backend.models import AdminReview, AuditLog, Complaint, DuplicateGroup, new_id, utcnow
from backend.responses import (AnalysisResponse, AnalyticsResponse, ComplaintPage, ComplaintView, GroupPage, GroupSplitResult,
                               GroupView, HealthResponse, StatsResponse, TrackingResponse)
from backend.schemas import (AdminLogin, AnalyzeRequest, ComplaintCreate, ComplaintPatch, DataDeletionRequest,
                             GroupConfirm, GroupSplit, Status, TokenResponse, validate_submission)
from backend.security import authenticate_admin, create_access_token, verify_access_token
from backend.retention import delete_complaints, purge_embedding_cache
from ml.location import LocationGazetteer
from ml.priority_rules import PriorityRules

LOG = logging.getLogger(__name__)
WRITE_LOCK_ID = 0x42414E474C415053


class RequestBoundary:
    """Bound request bodies, issue correlation IDs and avoid caching private data."""

    def __init__(self, app, max_body_bytes: int, allowed_origins=(), enforce_origins=False, production=False):
        self.app, self.maximum = app, max_body_bytes
        self.allowed_origins = {origin.rstrip("/") for origin in allowed_origins}
        self.enforce_origins, self.production = enforce_origins, production

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = new_id()
        scope.setdefault("state", {})["request_id"] = request_id

        started = False

        async def secure_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                csp = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
                if not self.production and scope.get("path") in ("/docs", "/redoc", "/docs/oauth2-redirect"):
                    csp = ("default-src 'none'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                           "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                           "img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; "
                           "font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode("ascii")),
                    (b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", csp.encode("ascii")), (b"cache-control", b"no-store")]
            await send(message)

        async def reject(code, detail):
            await JSONResponse({"detail": detail, "request_id": request_id}, status_code=code)(scope, receive, secure_send)

        headers = dict(scope.get("headers", []))
        if b"origin" in headers and self.enforce_origins:
            origin = headers[b"origin"].decode("latin-1")
            if origin.rstrip("/") not in self.allowed_origins:
                return await reject(403, "Origin is not allowed")
        if b"content-length" in headers:
            try:
                length = int(headers[b"content-length"])
                if length < 0:
                    raise ValueError()
            except ValueError:
                return await reject(400, "Invalid Content-Length")
            if length > self.maximum:
                return await reject(413, "Request body is too large")
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.maximum:
                return await reject(413, "Request body is too large")
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        try:
            await self.app(scope, replay, secure_send)
        except Exception as exc:
            if started:
                raise
            LOG.error("Service operation failed request_id=%s type=%s", request_id, type(exc).__name__)
            return await reject(503, "Service temporarily unavailable")


@contextmanager
def _write_transaction(factory):
    """Lock before any read, so matching and insertion see one stable corpus.

    PostgreSQL advisory locks are transaction-scoped and shared by all workers
    using this application. SQLite BEGIN IMMEDIATE provides the local analogue.
    External writers must use this protocol too; do not edit live tables by hand.
    """
    with factory() as session, session.begin():
        dialect = session.get_bind().dialect.name
        if dialect == "sqlite":
            session.execute(sql_text("BEGIN IMMEDIATE"))
        elif dialect == "postgresql":
            session.execute(sql_text("SELECT pg_advisory_xact_lock(:key)"), {"key": WRITE_LOCK_ID})
        else:
            raise RuntimeError("Unsupported write-lock dialect")
        yield session


def _index_records(session):
    fields = (Complaint.id, Complaint.text, Complaint.lang, Complaint.category, Complaint.location,
              Complaint.timestamp, Complaint.source_report_id)
    return [dict(row) for row in session.execute(select(*fields).where(Complaint.status != "rejected")).mappings()]


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


def _public_analysis(result):
    """A citizen may see their own analysis, never another submitter's metadata."""
    result = deepcopy(result)
    allowed = {"score", "threshold", "is_duplicate", "auto_group_eligible", "area_guard_status",
               "time_guard_status", "auto_group_block_reasons"}
    result["duplicate_candidates"] = [
        {key: value for key, value in item.items() if key in allowed}
        for item in result.get("duplicate_candidates", [])]
    return result


def _complaint_response(complaint, group_version=None, *, full=True, own=False):
    fields = ("id", "lang", "category", "severity", "priority_score", "priority_label", "status", "area_id",
              "group_id", "version", "created_at", "updated_at")
    result = {key: _iso(getattr(complaint, key)) for key in fields}
    result["group_version"] = group_version
    if full:
        result.update(text=complaint.text, location=complaint.location, timestamp=_iso(complaint.timestamp),
                      confidence=complaint.confidence, review_required=complaint.review_required,
                      analysis=_public_analysis(complaint.analysis) if own else deepcopy(complaint.analysis))
        if not own:
            result["contact"] = complaint.contact
    return result


def _group_response(session, group, *, full=True):
    count = session.scalar(select(func.count()).select_from(Complaint).where(Complaint.group_id == group.id))
    result = {"id": group.id, "status": group.status, "version": group.version, "size": count,
              "created_at": _iso(group.created_at), "updated_at": _iso(group.updated_at)}
    if full:
        result.update(confirmed_by=group.confirmed_by,
                      complaint_ids=list(session.scalars(select(Complaint.id).where(Complaint.group_id == group.id)
                                                        .order_by(Complaint.id).limit(1000))),
                      members_truncated=count > 1000)
    return result


def _review_values(complaint):
    return {key: getattr(complaint, key) for key in ("category", "severity", "priority_score", "priority_label",
                                                   "status", "review_required", "group_id", "version")}


def _stats_response(session):
    result = {"total_complaints": session.scalar(select(func.count()).select_from(Complaint)),
              "total_groups": session.scalar(select(func.count()).select_from(DuplicateGroup)),
              "needs_human_review": session.scalar(select(func.count()).select_from(Complaint).where(Complaint.review_required.is_(True)))}
    for key, field in (("category", Complaint.category), ("priority", Complaint.priority_label),
                       ("status", Complaint.status), ("area", Complaint.area_id), ("language", Complaint.lang)):
        result["by_" + key] = {str(value) if value is not None else "unknown": count
                              for value, count in session.execute(select(field, func.count()).group_by(field))}
    return result


def _audit(session, request, actor, action, entity_type, entity_id, changes):
    session.add(AuditLog(actor=actor, action=action, entity_type=entity_type, entity_id=entity_id,
                         changes=deepcopy(changes), request_id=request.state.request_id))


def _version(entity, expected):
    if entity.version != expected:
        raise HTTPException(409, "Stale version; fetch the current record and review again")


def _get(session, model, identifier):
    item = session.get(model, str(identifier))
    if item is None:
        raise HTTPException(404, "Record not found")
    return item


def create_app(settings: Settings | None = None, ml_service=None, engine=None) -> FastAPI:
    settings = (settings or Settings.from_env()).validate()
    owned_engine = engine is None
    engine = engine or make_engine(settings.database_url)
    factory = make_session_factory(engine)
    service = ml_service or MLService(settings)
    limiter = Limiter(key_func=get_remote_address, storage_uri=settings.rate_limit_storage_uri,
                      swallow_errors=False, headers_enabled=False)
    bearer = HTTPBearer(auto_error=False, description="Admin JWT from POST /auth/token")

    @asynccontextmanager
    async def lifespan(app):
        try:
            with engine.connect() as connection:
                tables = set(inspect(connection).get_table_names())
                if not {"complaints", "duplicate_groups", "admin_reviews", "audit_log"} <= tables:
                    raise RuntimeError("Backend tables missing; run python -m alembic upgrade head")
                if "contact" not in {column["name"] for column in inspect(connection).get_columns("complaints")}:
                    raise RuntimeError("Backend migration is not current; run python -m alembic upgrade head")
                if settings.environment == "production":
                    expected_heads = set(ScriptDirectory.from_config(AlembicConfig(str(ROOT / "alembic.ini"))).get_heads())
                    current_heads = set(MigrationContext.configure(connection).get_current_heads())
                    if current_heads != expected_heads:
                        raise RuntimeError("Backend migration is not current; run python -m alembic upgrade head")
            if not limiter._storage.check():
                raise RuntimeError("Rate-limit storage is unavailable")
            with factory() as session:
                service.initialize(_index_records(session))
            rules_path = Path(settings.rules_config)
            app.state.priority_rules = getattr(service.analyzer, "rules", None) or PriorityRules(
                rules_path if rules_path.is_absolute() else ROOT / rules_path)
            app.state.gazetteer = (getattr(service, "_guard_gazetteer", None) or
                                   getattr(service.analyzer, "gazetteer", None) or LocationGazetteer())
            yield
        finally:
            if owned_engine:
                engine.dispose()

    production = settings.environment == "production"
    app = FastAPI(title="BanglaPSG Complaints API", version="1.2.0", lifespan=lifespan,
                  docs_url=None if production else "/docs", redoc_url=None if production else "/redoc",
                  openapi_url=None if production else "/openapi.json",
                  description="Anonymous complaint submission and preview; JWT-protected administration. "
                              "Run migrations before startup. Complaint/group reviews require expected_version and reason; "
                              "privacy deletion previews before explicit application.")
    app.state.settings, app.state.engine = settings, engine
    app.state.session_factory, app.state.ml_service, app.state.limiter = factory, service, limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        LOG.error("Database operation failed request_id=%s type=%s", request.state.request_id, type(exc).__name__)
        return JSONResponse({"detail": "Database temporarily unavailable", "request_id": request.state.request_id}, status_code=503)

    @app.exception_handler(Exception)
    async def service_error(request, exc):
        # Do not log exception text: database/Redis URLs or complaint content may
        # appear inside third-party error messages. Correlate with request IDs.
        LOG.error("Service operation failed request_id=%s type=%s", request.state.request_id, type(exc).__name__)
        return JSONResponse({"detail": "Service temporarily unavailable", "request_id": request.state.request_id}, status_code=503)

    if settings.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=[value.rstrip("/") for value in settings.allowed_origins], allow_credentials=False,
                           allow_methods=["GET", "POST", "PATCH"], allow_headers=["Authorization", "Content-Type"],
                           expose_headers=["X-Request-ID"])
    app.add_middleware(RequestBoundary, max_body_bytes=settings.max_body_bytes,
                       allowed_origins=settings.allowed_origins,
                       enforce_origins=production or bool(settings.allowed_origins), production=production)

    def admin(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is not None and credentials.scheme.lower() == "bearer":
            try:
                return verify_access_token(credentials.credentials, settings)
            except ValueError:
                pass
        raise HTTPException(401, "Valid administrator token required", headers={"WWW-Authenticate": "Bearer"})

    def reader(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None and settings.allow_public_reads:
            return None
        return admin(credentials)

    def refresh_after_commit():
        try:
            with factory() as session:
                service.refresh(_index_records(session))
        except Exception as exc:
            # The transaction committed. Do not return a failure that induces a
            # retry/double submission. The service blocks stale-index analysis
            # until a subsequent refresh succeeds.
            LOG.error("Committed write; index refresh failed type=%s", type(exc).__name__)

    def validated_record(payload):
        try:
            text = validate_submission(payload.text, settings.max_text_length, settings.abuse_max_urls)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"id": new_id(), "text": text, "lang": payload.lang, "location": payload.location or "",
                "timestamp": _iso(payload.timestamp) or ""}

    @app.post("/auth/token", response_model=TokenResponse, tags=["Admin authentication"])
    @limiter.limit(settings.login_rate_limit)
    def login(request: Request, payload: AdminLogin):
        """Exchange configured admin credentials for a short-lived bearer JWT."""
        if not authenticate_admin(payload.username, payload.password, settings):
            raise HTTPException(401, "Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
        token, seconds = create_access_token(settings.admin_username, settings)
        return TokenResponse(access_token=token, expires_in=seconds)

    @app.post("/analyze", tags=["Citizen"], response_model=AnalysisResponse, response_model_exclude_unset=True)
    @limiter.limit(settings.analysis_rate_limit)
    def analyze(request: Request, payload: AnalyzeRequest):
        """Preview analysis without writing complaints, reviews or audit records."""
        record = validated_record(payload)
        with service.lock, factory() as session:
            return _public_analysis(service.analyze(record, existing_records=_index_records(session)))

    @app.post("/complaints", status_code=201, tags=["Citizen"], response_model=ComplaintView, response_model_exclude_unset=True)
    @limiter.limit(settings.submission_rate_limit)
    def submit(request: Request, payload: ComplaintCreate):
        """Persist a report and propose membership only after area/time guards."""
        record = validated_record(payload)
        with service.lock:
            with _write_transaction(factory) as session:
                result = service.analyze(record, existing_records=_index_records(session))
                identifiers = [item["id"] for item in result.get("duplicate_candidates", []) if item.get("auto_group_eligible")]
                group_ids = set(session.scalars(select(Complaint.group_id).where(Complaint.id.in_(identifiers),
                                                                                Complaint.group_id.is_not(None)))) if identifiers else set()
                group = session.get(DuplicateGroup, next(iter(group_ids))) if len(group_ids) == 1 else None
                if group is not None and group.status != "reviewed_split":
                    group.version += 1
                    group.updated_at, group.status, group.confirmed_by = utcnow(), "proposed", None
                else:
                    if group_ids:
                        result.update(review_required=True, review_status="needs_human_review",
                                      review_reasons=list(dict.fromkeys(result.get("review_reasons", []) +
                                                                       ["duplicate_group_ambiguity_or_admin_partition"])))
                    group = DuplicateGroup(id=new_id())
                    session.add(group)
                    session.flush()
                complaint = Complaint(id=record["id"], text=record["text"], lang=result["lang"],
                                      category=result.get("category"), severity=result.get("severity"),
                                      priority_score=result["priority_score"], priority_label=result["priority_label"],
                                      confidence=result.get("confidence"), review_required=bool(result.get("review_required", True)),
                                      status="in_review" if result.get("review_required", True) else "submitted",
                                      area_id=result.get("area_id"), location=payload.location, contact=payload.contact, timestamp=payload.timestamp,
                                      group_id=group.id, source_report_id=record["id"], analysis=deepcopy(result))
                session.add(complaint)
                session.flush()
                _audit(session, request, "anonymous", "complaint_submit", "complaint", complaint.id,
                       {"status": complaint.status, "group_id": group.id, "area_id": complaint.area_id,
                        "review_required": complaint.review_required})
                response = _complaint_response(complaint, group.version, own=True)
            refresh_after_commit()
        return response

    @app.get("/complaints", tags=["Complaints"], response_model=ComplaintPage, response_model_exclude_unset=True)
    def list_complaints(category: str | None = Query(None, max_length=128),
                        priority: str | None = Query(None, max_length=32),
                        area: str | None = Query(None, max_length=256), status: Status | None = None,
                        group_id: UUID | None = None, review_required: bool | None = None, page: int = Query(1, ge=1),
                        page_size: int = Query(25, ge=1, le=100), actor=Depends(reader)):
        """Filter and paginate reports. Reads require JWT unless redacted reads are enabled."""
        conditions = []
        if category is not None:
            conditions.append(func.lower(Complaint.category) == category.lower())
        if priority is not None:
            conditions.append(Complaint.priority_label == priority)
        if status is not None:
            conditions.append(Complaint.status == status)
        if group_id is not None:
            conditions.append(Complaint.group_id == str(group_id))
        if review_required is not None:
            conditions.append(Complaint.review_required.is_(review_required))
        if area is not None:
            context = app.state.gazetteer.extract("", area)
            canonical = context.get("canonical_id")
            conditions.append(Complaint.area_id == canonical if canonical else
                              or_(Complaint.area_id == area, func.lower(Complaint.location) == area.lower()))
        with factory() as session:
            total = session.scalar(select(func.count()).select_from(Complaint).where(*conditions))
            rows = session.execute(select(Complaint, DuplicateGroup.version).outerjoin(DuplicateGroup, Complaint.group_id == DuplicateGroup.id)
                                   .where(*conditions).order_by(Complaint.priority_score.desc(), Complaint.created_at.desc(), Complaint.id)
                                   .offset((page - 1) * page_size).limit(page_size))
            return {"items": [_complaint_response(row, version, full=actor is not None) for row, version in rows],
                    "total": total, "page": page, "page_size": page_size}

    @app.get("/complaints/{complaint_id}", tags=["Complaints"], response_model=ComplaintView, response_model_exclude_unset=True)
    def get_complaint(complaint_id: UUID, actor=Depends(reader)):
        with factory() as session:
            complaint = _get(session, Complaint, complaint_id)
            group = session.get(DuplicateGroup, complaint.group_id) if complaint.group_id else None
            return _complaint_response(complaint, group.version if group else None, full=actor is not None)

    @app.get("/tracking/{complaint_id}", tags=["Citizen"], response_model=TrackingResponse)
    @limiter.limit(settings.analysis_rate_limit)
    def tracking(request: Request, complaint_id: UUID):
        """Return minimal status for a random tracking ID; never disclose report/contact content.

        The ID is a capability: keep it private. There is deliberately no public
        tracking directory/search. Anonymous users cannot retrieve full reports.
        """
        with factory() as session:
            complaint = _get(session, Complaint, complaint_id)
            return {field: _iso(getattr(complaint, field)) for field in (
                "id", "status", "category", "severity", "priority_score", "priority_label", "review_required",
                "area_id", "lang", "created_at", "updated_at")}

    @app.patch("/complaints/{complaint_id}", tags=["Admin reviews"], response_model=ComplaintView, response_model_exclude_unset=True)
    def patch_complaint(request: Request, complaint_id: UUID, payload: ComplaintPatch, actor=Depends(admin)):
        """Confirm/override current fields; preserve immutable automatic analysis as feedback provenance."""
        changes = payload.model_dump(exclude_unset=True, exclude={"expected_version", "reason"})
        rules = app.state.priority_rules
        if "priority_label" in changes and changes["priority_label"] not in {band["label"] for band in rules.priority_bands}:
            raise HTTPException(422, "Priority label is not configured in priority_rules.yaml")
        if "priority_score" in changes and "priority_label" not in changes:
            changes["priority_label"] = rules.label_for_score(changes["priority_score"])
        with service.lock:
            with _write_transaction(factory) as session:
                complaint = _get(session, Complaint, complaint_id)
                _version(complaint, payload.expected_version)
                before = _review_values(complaint)
                for key, value in changes.items():
                    setattr(complaint, key, value)
                complaint.review_required = complaint.status == "in_review"
                complaint.version += 1
                complaint.updated_at = utcnow()
                after = _review_values(complaint)
                session.add(AdminReview(complaint_id=complaint.id, actor=actor, action="complaint_update", reason=payload.reason,
                                        previous_values=before, new_values=after))
                _audit(session, request, actor, "complaint_update", "complaint", complaint.id,
                       {"reason": payload.reason, "previous_values": before, "new_values": after})
                group = session.get(DuplicateGroup, complaint.group_id) if complaint.group_id else None
                response = _complaint_response(complaint, group.version if group else None)
            refresh_after_commit()
        return response

    @app.get("/duplicate-groups", tags=["Duplicate groups"], response_model=GroupPage, response_model_exclude_unset=True)
    def list_groups(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
                    group_id: UUID | None = None, actor=Depends(reader)):
        with factory() as session:
            conditions = [DuplicateGroup.id == str(group_id)] if group_id else []
            total = session.scalar(select(func.count()).select_from(DuplicateGroup).where(*conditions))
            groups = session.scalars(select(DuplicateGroup).where(*conditions).order_by(DuplicateGroup.updated_at.desc(), DuplicateGroup.id)
                                     .offset((page - 1) * page_size).limit(page_size)).all()
            return {"items": [_group_response(session, group, full=actor is not None) for group in groups],
                    "total": total, "page": page, "page_size": page_size}

    @app.get("/duplicate-groups/{group_id}", tags=["Duplicate groups"], response_model=GroupView)
    def get_group(group_id: UUID, actor=Depends(admin)):
        with factory() as session:
            return _group_response(session, _get(session, DuplicateGroup, group_id))

    @app.post("/duplicate-groups/{group_id}/confirm", tags=["Admin reviews"], response_model=GroupView, response_model_exclude_unset=True)
    def confirm_group(request: Request, group_id: UUID, payload: GroupConfirm, actor=Depends(admin)):
        """Confirm membership, optionally moving explicitly reviewed complaint IDs into this group."""
        with service.lock:
            with _write_transaction(factory) as session:
                group = _get(session, DuplicateGroup, group_id)
                _version(group, payload.expected_version)
                before = _group_response(session, group)
                moved = []
                # Resolve all IDs first. Any missing record rolls back every change.
                complaints = [_get(session, Complaint, identifier) for identifier in payload.complaint_ids or []]
                changed_groups = set()
                for complaint in complaints:
                    if complaint.group_id == group.id:
                        continue
                    old_id = complaint.group_id
                    moved.append({"complaint_id": complaint.id, "previous_group_id": old_id})
                    complaint.group_id, complaint.updated_at = group.id, utcnow()
                    complaint.version += 1
                    if old_id and old_id not in changed_groups:
                        old = _get(session, DuplicateGroup, old_id)
                        old.version += 1
                        # A reviewed membership move does not erase the distinct
                        # incident boundary previously established for remaining
                        # members of the source group.
                        old.updated_at, old.confirmed_by = utcnow(), None
                        if old.status != "reviewed_split":
                            old.status = "proposed"
                        changed_groups.add(old_id)
                    _audit(session, request, actor, "group_member_move", "complaint", complaint.id,
                           {"previous_group_id": old_id, "group_id": group.id, "reason": payload.reason})
                group.version += 1
                group.status, group.confirmed_by, group.updated_at = "confirmed", actor, utcnow()
                session.flush()
                response = _group_response(session, group)
                after = {**response, "moved_complaints": moved}
                session.add(AdminReview(group_id=group.id, actor=actor, action="group_confirm", reason=payload.reason,
                                        previous_values=before, new_values=after))
                _audit(session, request, actor, "group_confirm", "duplicate_group", group.id,
                       {"reason": payload.reason, "previous_values": before, "new_values": after})
            refresh_after_commit()
        return response

    @app.post("/duplicate-groups/{group_id}/split", tags=["Admin reviews"], response_model=GroupSplitResult, response_model_exclude_unset=True)
    def split_group(request: Request, group_id: UUID, payload: GroupSplit, actor=Depends(admin)):
        """Move a proper subset to a separate reviewed partition; automation cannot undo it."""
        with service.lock:
            with _write_transaction(factory) as session:
                group = _get(session, DuplicateGroup, group_id)
                _version(group, payload.expected_version)
                members = session.scalars(select(Complaint).where(Complaint.group_id == group.id)).all()
                member_ids, selected = {item.id for item in members}, set(payload.complaint_ids)
                if not selected < member_ids:
                    raise HTTPException(422, "Split IDs must be a nonempty proper subset of current group members")
                before = _group_response(session, group)
                new_group = DuplicateGroup(id=new_id(), status="reviewed_split")
                session.add(new_group)
                session.flush()
                for complaint in members:
                    if complaint.id in selected:
                        complaint.group_id = new_group.id
                    complaint.version += 1
                    complaint.updated_at = utcnow()
                group.version += 1
                group.status, group.confirmed_by, group.updated_at = "reviewed_split", None, utcnow()
                session.flush()
                response = {"original_group": _group_response(session, group), "new_group": _group_response(session, new_group)}
                session.add(AdminReview(group_id=group.id, actor=actor, action="group_split", reason=payload.reason,
                                        previous_values=before, new_values=deepcopy(response)))
                _audit(session, request, actor, "group_split", "duplicate_group", group.id,
                       {"reason": payload.reason, "previous_values": before, "new_values": response})
            refresh_after_commit()
        return response

    @app.get("/stats", tags=["Operations"], response_model=StatsResponse)
    def stats():
        """Aggregate counts only; never expose complaint wording or admin credentials."""
        with factory() as session:
            return _stats_response(session)

    @app.post("/admin/data-deletion", tags=["Admin privacy maintenance"])
    def privacy_deletion(request: Request, payload: DataDeletionRequest, actor=Depends(admin)):
        """Preview by default. Explicit apply deletes selected reports and sanitizes linked copies.

        Backups, exported files, other worker memory and separately trained models
        require the operator's maintenance runbook; this is not machine unlearning.
        """
        with service.lock:
            with _write_transaction(factory) as session:
                if not payload.apply:
                    found = set(session.scalars(select(Complaint.id).where(Complaint.id.in_(payload.complaint_ids))))
                    return {"dry_run": True, "would_delete_ids": sorted(found),
                            "missing_ids": sorted(set(payload.complaint_ids) - found), "deleted_count": 0}
                result = delete_complaints(session, payload.complaint_ids, actor, payload.reason,
                                           request_id=request.state.request_id)
                if result["deleted_count"]:
                    cache_path = Path(settings.embedding_cache_dir)
                    result["cache_cleanup"] = purge_embedding_cache(cache_path if cache_path.is_absolute() else ROOT / cache_path)
            refresh_after_commit()
        return {"dry_run": False, **result}

    @app.get("/analytics", tags=["Operations"], response_model=AnalyticsResponse)
    def analytics(actor=Depends(admin)):
        """Administrator chart aggregates across all rows, independent of queue pagination.

        Daily counts describe current priorities by report creation date. They
        are not a history of past model decisions or resolution times.
        """
        with factory() as session:
            result = _stats_response(session)
            created = Complaint.created_at
            if session.get_bind().dialect.name == "postgresql":
                created = func.timezone("UTC", created)
            day = func.date(created)
            result["priority_over_time"] = [
                {"date": str(value), "priority": priority, "count": count}
                for value, priority, count in session.execute(
                    select(day, Complaint.priority_label, func.count()).group_by(day, Complaint.priority_label)
                    .order_by(day, Complaint.priority_label))]
            result["review_workload"] = [
                {"status": status, "review_required": required, "count": count}
                for status, required, count in session.execute(
                    select(Complaint.status, Complaint.review_required, func.count())
                    .group_by(Complaint.status, Complaint.review_required).order_by(Complaint.status, Complaint.review_required))]
            return result

    @app.get("/health", tags=["Operations"], response_model=HealthResponse, response_model_exclude_unset=True)
    def health():
        """Readiness is degraded in development when trained artifacts are absent."""
        try:
            with factory() as session:
                session.execute(select(Complaint.id).limit(1)).first()
            storage_ok = bool(limiter._storage.check())
            ml = service.readiness()
            if not storage_ok:
                return JSONResponse({"status": "unavailable", "database": "ok", "rate_limit_storage": "unavailable"}, status_code=503)
        except Exception:
            return JSONResponse({"status": "unavailable", "detail": "A required dependency is unavailable"}, status_code=503)
        # No artifact paths, database URLs, secrets or third-party messages.
        public_ml = {key: value for key, value in ml.items() if key != "messages"}
        return JSONResponse({"status": "ready" if ml["ready"] else "degraded", "database": "ok",
                             "rate_limit_storage": "ok", "ml": public_ml},
                            status_code=503 if settings.require_ml_ready and not ml["ready"] else 200)

    return app
