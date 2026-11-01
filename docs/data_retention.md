# Reference data retention and deletion policy

Policy version: 2026-10-31. Adopt and staff this operational policy before using
the portal with real people. The repository includes a working operator CLI and
JWT-protected administration endpoint for reviewed deletion; it does not
install an automatic retention schedule or establish that legal requirements
have been met. Until the operator supplies its public name/contact and adopts
the policy, the `/privacy` page identifies the deployment as a demonstration.

## Ownership, purpose and minimisation

The organisation operating the deployment owns the retention decisions and
request channel. Configure its real public name and contact in the frontend
environment, assign a trained private data steward, and document hosting,
database, Redis and backup processors/regions. Collect only information needed
to assess public-service grievances. Phone/email is optional; the portal sends
no automatic notification and must not imply one. Do not require a citizen
account, identity-document upload, or unnecessary third-party personal details.

Complaint data supports service review, related-incident investigation and
status tracking. It is not automatically licensed or consented research data.
Research reuse needs separate approval/permission, minimisation, disclosure of
derived versions, access controls and a deliberate evaluation cohort policy.

## Reference schedule

The periods below are proposed operating limits, **not experiment results**.
An operator adopting this notice must be able to perform them; otherwise revise
the policy/notice transparently before accepting personal information.

| Data | Reference maximum / action | Operator responsibility |
| --- | --- | --- |
| Closed complaint text, raw location, contact and stored automatic analysis | Default deletion eligibility: resolved/rejected records last updated more than 90 days ago | Run reviewed retention batches regularly; the implementation uses `updated_at`, not an unavailable historical closure timestamp. The operator must publish and meet its actual erasure schedule. |
| Open complaints | Review every 180 days | Confirm necessity, resolve or minimise; document any continued need and next review date. |
| Admin-review/audit narratives and before/after JSON | Up to 180 days after the related case closes | Minimise sensitive reasons and analysis terms; retain only proportionate accountability evidence. |
| Infrastructure diagnostic/access logs | Up to 30 days where controlled by the operator | Disable unnecessary body/URL logging and configure each provider's retention. Document shorter/provider-mandated limits. |
| Temporary feedback, prediction/error CSVs and operational exports | Up to 30 days | Move only separately approved, minimised research records into the governed research dataset; delete unmanaged copies. |
| Rolling backups | Expire within 30 days | Encrypt, restrict access, and keep a deletion ledger to apply again after a restore. |
| Derived embeddings/cache/index files | No longer than their source records | Rebuild or purge every affected cache/index, including copied worker volumes and private artifact transfers. |

Document a dispute, appeal or applicable legal hold with scope, justification,
authorised owner and review/expiry date. A hold is not an indefinite default.
An operational metric aggregated before deletion may be retained only if the
operator checks re-identification risk; no raw text or contact belongs in a
public thesis result bundle.

## Requests from a complainant

1. Receive a request through the published operator contact. Ask for the
   tracking ID and the requested correction/deletion scope. Possession of an ID
   alone is not proof that the requester owns every report in a duplicate group.
2. Acknowledge within 7 business days. Use proportionate verification agreed
   with the requester; do not routinely demand new identity documents or reveal
   private complaint text to an unverified person.
3. Aim to decide within 30 days. Explain scope, limitations, holds and next
   steps; apply any stricter applicable deadline. These targets require a staffed
   process and are not implemented by the API automatically.
4. Record the minimum request/action evidence in a restricted deletion ledger.
   Keep that ledger outside research exports and avoid copying the original
   grievance or contact details into it.

## Working deletion interfaces

There is no anonymous deletion endpoint. A verified request is handled by an
authorised administrator/data steward. Supplying a tracking ID does not bypass
authentication or establish ownership; perform the proportionate request
verification above before choosing a record for deletion.

### Operator CLI: dry-run first

Run from the repository root, with database/cache settings in the private
environment or root `.env`. The script never accepts a database password as a
command-line argument or prints the database URL.

```powershell
# Read-only selection: currently resolved/rejected, updated more than 90 days ago.
python scripts/delete_data.py

# A specific verified request: repeat --complaint-id for multiple records.
python scripts/delete_data.py --complaint-id "<verified tracking UUID>"

# A different approved closed-case retention window:
python scripts/delete_data.py --retention-days 120 --limit 100

# Explicit UTC cutoff; future or timezone-free cutoffs are rejected.
python scripts/delete_data.py --older-than "2026-01-01T00:00:00Z"
```

The UUID above stands for the actual ID reviewed by the operator, not a usable
sample target. Dry-run receipts contain private candidate IDs and counts, not
complaint/contact content. They do not modify domain rows or purge caches.

Before applying, stop all API workers for the maintenance window, review the
candidate list against the deletion ledger and any valid hold, and use explicit
approved IDs where some candidates must be retained. There is no database
legal-hold field or automated ownership/hold adjudication. The current schema
does not contain an immutable `closed_at`: any later review postpones the
last-update-based retention eligibility. Open complaints are never selected by
the age selector; a separately verified privacy request can explicitly select
one. Retention batches are capped at 1,000 records; rerun reviewed batches as
needed, rather than claiming one run erased an unbounded corpus.

```powershell
python scripts/delete_data.py --complaint-id "<verified tracking UUID>" --apply --actor "authorised-steward" --reason "verified_privacy_request"
```

`--apply`, `--actor` and `--reason` are required for erasure. Use a short
non-personal rationale, since shell history/process listings may expose command
arguments. The free-text rationale is deliberately not copied into the new
database audit event; the protected deletion ledger records the minimum request
decision. Keep the resulting private receipt. Restart all API workers before
reopening service so old in-memory indexes are discarded.

### Administrator API: explicit application

`POST /admin/data-deletion` requires a valid administrator bearer JWT. First
submit the reviewed IDs with the default `apply: false` to inspect the scope:

```json
{
  "complaint_ids": ["<verified tracking UUID>"],
  "reason": "verified_privacy_request",
  "apply": false
}
```

After the scope/request review, send the same request with `"apply": true`.
This is an irreversible administration action, not a citizen self-service form.
The server uses the ML lock and coordinated database transaction; cache failure
prevents a successful domain-erasure commit. The active worker's index is
refreshed after commit. Restart every other worker for immediate in-memory
erasure, and inventory separate cache volumes/replicas as described below.
The API does not perform ownership checks on behalf of the human administrator,
adjudicate holds or remove external backups/exports.

## Exact scope of the implementation

- Target `complaints` rows are physically deleted, including text, raw location,
  contact and the stored automatic analysis. Existing read/tracking routes then
  return not found for those IDs. Missing IDs are reported idempotently.
- Direct complaint reviews are deleted before their restrictive foreign keys
  can block erasure. Known related/historical group reviews have complete
  narratives and before/after JSON redacted and are excluded from future feedback
  exports. This can remove other narrative context in the same reviewed group;
  their other citizens' actual complaint rows are retained.
- Known related audit JSON, source identifiers and old request correlations are
  removed. Actor/action/time accountability remains. A new deletion event stores
  the authorised actor, action, time and aggregate counts, without raw request
  rationale, complaint IDs, complaint text or contact details.
- Surviving analyses in affected groups or referencing deleted IDs are cleared
  and marked for review, preventing old duplicate candidates and stale group
  evidence from exposing removed identifiers. Their own submitted content is
  retained. Reviewed split boundaries remain protected. Empty group records are
  retained where needed for audit/review references.
- A remaining `source_report_id` reference is cleared and reported as a lineage
  warning. The function does not silently delete additional reports. Review
  derived language copies separately and include their approved IDs in the
  request where erasure scope requires it.
- The configured cache purge operates only on its fixed `embeddings.sqlite3`
  file after checking the exact cache schema. It clears embeddings/namespaces
  and compacts/checkpoints the database; it does not recursively delete folders,
  models or unrelated databases. All vectors in that cache are cleared because
  cleaned-text keys can be shared across reports. Remaining reports rebuild their
  cache as needed. Links/junctions and unexpected schemas are rejected.

Erasure runs inside an explicit database transaction. A failed cache purge
rolls back the domain erasure; an already emptied cache can be rebuilt. Cache
compaction is not a forensic guarantee about SSD wear levelling, operating-system
snapshots or provider storage. A maintenance window prevents other workers from
simultaneously rewriting the cache. For a separate copy/cache volume, repeat the
approved purge there and restart the corresponding workers.
Physical row deletion here means `DELETE` from the live relational tables, not
certified media wiping. Database journal/free-page/MVCC storage and provider
recovery snapshots require the operator's database maintenance and backup-expiry
controls; do not promise immediate destruction of every underlying storage byte.

## Copies that still require operator action

1. Inventory private feedback/prediction/error exports, translations, screenshots,
   replicas and manually copied review text. Identifier-based cleanup cannot
   discover arbitrary unlinked copies. Remove or irreversibly redact authorised
   derived/export copies and revoke unmanaged access.
2. Verify post-erasure full admin reads, tracking, group views and fresh feedback
   exports against the request scope. Keep a minimal protected deletion ledger
   for later backup restoration, without copying the erased personal content.
3. Let protected backups expire under the published schedule; prevent ordinary
   access to the removed content. After a restore, replay the deletion ledger
   before serving requests. Explain this limited backup window to the requester.
4. If governed research datasets or model versions contained the information,
   assess retraining/model withdrawal under the research consent and governance
   process. Database deletion does not establish machine unlearning.

Do not use ad-hoc blanket SQL or `docker compose down -v` as a privacy process.
The latter can destroy unrelated complaints while leaving other copies intact.
Test this runbook on synthetic records and verify constraints and the applicable
backup/hold decisions before acting on live personal information.

## Publication and review

Publish the bilingual `/privacy` route and link it from submission/footer.
Changing the retention schedule requires corresponding notice updates and an
operator decision. Review the policy at each deployment change and after a
privacy incident. The app's source checks are not evidence that a particular
operator has performed retention, deletion, notification or consent obligations.
