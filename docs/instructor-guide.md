# Instructor Guide

## Before the day

- [ ] Create the workshop catalog and grant attendees `USE CATALOG` + `CREATE SCHEMA` (see the [README](../README.md#-prerequisites)).
- [ ] Push this repo to a Git provider the workspace can reach. Check that a test user can create a **Git folder** from it.
- [ ] Confirm attendees can choose **serverless environment version 4+** (needed for Spark ML in Module 04)
- [ ] Confirm **serverless notebooks** are enabled and that attendees can `%pip install` from PyPI (or from your mirror).
- [ ] Do a full dry run as a non-admin user: run `notebooks/00` → `08` in order on serverless. This takes about 25 minutes, mostly waiting for the monitor refresh.
- [ ] Confirm attendees have **CAN USE** on a SQL warehouse (serverless preferred). Module 10 reads traces from Unity Catalog through it.
- [ ] *(Module 07 demo)* Pre-create one serving endpoint the day before, so you can show it without waiting about 10 minutes live.
- [ ] *(Optional, Module 10 AutoML demo)* AutoML classification and regression don't run on serverless (only forecasting does). To demo AutoML on the churn data, start a classic cluster with **Databricks Runtime ML** in **dedicated (single user) access mode** ahead of time.
- [ ] *(Module 09 demo)* Create `ml_workshop_staging` / `ml_workshop_prod` if you want to deploy those targets live.

## Timing and delivery mode

| Module | Time | Mode | If you're behind |
|---|---|---|---|
| 00 Setup | 15m | Hands-on | Have the complete `00_setup` notebook ready for anyone stuck |
| 01 Whole game | 30m | Hands-on + discussion | Don't cut this; it sets up the whole day |
| 02 Lakehouse data | 45m | Hands-on | Skip Cell 4 (SQL) and the exercise |
| 03 MLflow | 45m | Hands-on | Reduce Optuna to 10 trials |
| 04 Pipelines | 45m | Hands-on | Run Cell 7 (Spark save/load) as a demo |
| 05 Scale out | 35m | Hands-on or demo | Demo Cells 2–4, let attendees run 5–6 |
| 06 Feature Store | 40m | Hands-on | Skip the exercise |
| 07 Registry | 30m | Hands-on (Cell 7 demo) | — |
| 08 Monitoring | 40m | Hands-on | Start Cell 5 *before* the concept talk so the refresh runs while you explain PSI |
| 09 Bundles | 35m | Demo + hands-on dev deploy | Pure demo; attendees deploy after the workshop |
| 10 Wrap-up | 15m | Discussion | — |

## The "aha" moment for each module

Each module has one moment the room should not miss. Pause there.

- **01**: the dummy classifier ties our model on accuracy. *"Would you have presented 81% to your boss?"* Then build the 8-questions list **with the room** before showing the table.
- **02**: Cell 2 returns instantly on any data size, and `explain()` shows the plan. Then the lineage graph, which nobody drew.
- **03**: open the experiment UI and compare runs, then show the **dataset version** on a run. That's the auditor's question answered.
- **04**: the Medan-only batch crashes the legacy model and then works with the pipeline. Ask *"what if it hadn't crashed?"*
- **05**: the per-region models are all *worse* than the global one. That's a cheap experiment instead of a debate.
- **06**: the scoring DataFrame has **no transaction columns**, and predictions still work.
- **07**: rollback is one line. Ask the room how long a rollback takes on their current platform.
- **08**: the predicted churn rate barely moves while the actual rate jumps. *"This is your model in production today, and nobody is measuring it."*
- **09**: twenty people deploy the same bundle and nobody collides. Then show the Git history as the change log.

## Common issues

| Symptom | Cause / fix |
|---|---|
| `PERMISSION_DENIED` on `CREATE SCHEMA` | Attendee lacks `CREATE SCHEMA` on the catalog; re-run the `GRANT` |
| `FileNotFoundError` in Module 00 Cell 2 | Repo cloned under a different name or path; fix `DATA_DIR`, or upload the CSVs to the `raw` volume through the UI |
| `Py4JSecurityException: Constructor ... Imputer ... is not whitelisted` (Module 04) | Serverless environment version is below 4; set it in the notebook's **Environment** panel and re-run |
| `Tag value ... is not an allowed value for tag policy key ...` (Module 07) | The workspace has a **governed tag policy** on that key; pick a different tag key or an allowed value |
| `AttributeError: Can't get attribute '_RemainderColsList'` (or similar) when loading a model | scikit-learn version differs from the one that trained the model; keep the pinned `scikit-learn==1.6.1` in every notebook |
| `A UC Trace Destination can only be linked to an Experiment that does not already contain any traces` (Module 03 Cell 1) | The `churn_workshop` experiment already has traces in the old storage, typically from an earlier dry run of Module 10. Rename or delete the experiment under **Experiments**, then re-run the cell |
| `SQL warehouse ID is required for accessing traces in UC tables` (Module 10) | `MLFLOW_TRACING_SQL_WAREHOUSE_ID` isn't set, or the attendee can't see any warehouse; grant **CAN USE** on one, or paste its ID into the config cell |
| `Encountered unexpected error during resolving tags: getContext().extraContext() is not a CommandContext accessor` (every MLflow run) | Harmless. MLflow tries to read Git details (repo URL, commit) for notebooks in a Git folder, and some workspaces, such as **Free Edition**, don't expose them. Runs are logged normally, just without the Git tags |
| `NameError` after `%pip install` | `restartPython()` clears variables; re-run the config cell |
| `IndexError` / empty result in Module 05 Cell 5 or Module 07 Cell 2 | Module 04 Cell 5 wasn't run (no run tagged `candidate=true`) |
| Module 07 Cell 3 can't find `hist_gbm_tuning` | Module 03 Cell 6 wasn't run; set `best` by hand: `{"learning_rate": 0.06, "max_depth": 3, "max_iter": 300, "l2_regularization": 0.5}` |
| Module 08 Cell 6 waits a long time | First refresh takes 5–10 min; continue with the concept discussion, or move on to Cell 8 and come back |
| `fe.log_model` raises a `TypeError` about unexpected arguments | Version mismatch between `databricks-feature-engineering` and MLflow; `%pip install -U databricks-feature-engineering` |
| Bundle deploy: `schema` variable has no value | Deploying a target other than `dev` without that target's variables; check `databricks.yml` |
| Module 09 Step 3 `StopIteration` | The bundle hasn't been deployed to `dev` yet, or it was deployed by another user |

## Talking points on "why not just keep the legacy platform?"

Keep it about **capabilities, not features**. The legacy platform can train a model. What it can't do is answer the eight production questions *without heroics*. Every module ends with a *Legacy vs Databricks* table, and the [cheat sheet](cheat-sheet.md) collects them all onto one page you can hand out.
