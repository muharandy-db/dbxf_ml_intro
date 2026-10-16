# Module 09 — From Notebook to Production (Jobs + Databricks Asset Bundles)

> 🎯 **Learning objectives**
> - Explain why "copy the notebook to the prod workspace" fails, and what **infrastructure-as-code** fixes
> - Read a **Databricks Asset Bundle**: bundle, resources, variables, **targets** (dev / staging / prod)
> - Deploy and run the churn job in **your dev target**, and see how the same files promote to staging and prod
>
> ⏱ **Time:** 35 minutes  ·  Answers production question **#8**: *How do we move this from my notebook to production safely?*

---

## 🏚 The legacy way

Going to production means: export the notebook, import it into the production environment, find-and-replace `dev_db` with `prod_db`, set up the scheduler by clicking through a UI, and hope the cluster settings match. Staging is "mostly the same as prod, except for the bits someone changed by hand." Nobody can answer *"what exactly is deployed in prod, and who changed it?"* without logging in and looking around.

---

## 💡 The idea: one recipe, three kitchens

A restaurant chain doesn't let each branch improvise the menu. There is one **recipe book**, and each branch follows it with its own local suppliers. Change the recipe book, and every branch changes the same way.

A **Databricks Asset Bundle (DAB)** is that recipe book for your project. It's a folder in Git with:

- your **code**: the training and scoring scripts
- a **`databricks.yml`** describing the **resources** that code needs (jobs, schedules, compute environments, permissions, even models and experiments) as YAML
- **variables** for everything that differs between environments (catalog, schema, workspace)
- **targets**, the named environments (`dev`, `staging`, `prod`), each setting those variables differently

`databricks bundle deploy -t <target>` makes the workspace match the files. The job definition, its schedule, and its Python environment are created or updated to match, the same way every time. Because it lives in Git, every change to production is a **reviewed pull request** with a history.

Two details make it practical for teams:

1. **`mode: development`** prefixes everything you deploy with `[dev your_name]` and pauses schedules, so twenty attendees can deploy the same bundle to the same workspace without collisions.
2. **`mode: production`** adds guard rails. It's meant to be deployed by **CI/CD as a service principal**, not from someone's laptop.

The one caveat: a bundle describes *what* to deploy, but not *whether* the model is good enough. That decision still comes from evidence, which is why `train.py` contains the champion/challenger check from Module 07.

---

## 🧭 What's in `bundle/`

```text
bundle/
├── databricks.yml            # bundle name, variables, targets
├── resources/
│   └── churn_job.yml         # the job: train_and_evaluate → batch_score
└── src/
    ├── train.py              # Modules 04 + 07 as one parameterised script
    └── batch_score.py        # score the latest snapshot with @champion
```

### `databricks.yml`: variables and targets

```yaml
variables:
  catalog:
    default: ml_workshop
  schema:
    description: Schema inside the catalog
  model_name:
    default: churn_model

targets:
  dev:
    mode: development
    default: true
    variables:
      schema: ${workspace.current_user.short_name}   # your personal schema from Module 00
  staging:
    variables:
      catalog: ml_workshop_staging
      schema: churn
  prod:
    mode: production
    variables:
      catalog: ml_workshop_prod
      schema: churn
```

**What to point out**
- **Same code, different data.** In dev the job reads and writes `ml_workshop.<you>`, which is exactly the schema you've used all day. In prod it reads and writes `ml_workshop_prod.churn`. Nobody runs find-and-replace.
- Targets can also point at **different workspaces** (`workspace.host`), which is common for regulated production environments.

### `resources/churn_job.yml`: the job

```yaml
resources:
  jobs:
    churn_training:
      name: churn-training-${bundle.target}
      tasks:
        - task_key: train_and_evaluate
          environment_key: ml
          spark_python_task:
            python_file: ../src/train.py
            parameters: [--catalog=${var.catalog}, --schema=${var.schema}, ...]
        - task_key: batch_score
          depends_on: [{ task_key: train_and_evaluate }]
          ...
      environments:
        - environment_key: ml
          spec:
            environment_version: "4"
            dependencies: [mlflow>=3.1, scikit-learn==1.6.1]
      schedule:
        quartz_cron_expression: "0 0 6 ? * MON"
        timezone_id: Asia/Jakarta
```

**What to point out**
- Two tasks with a **dependency**: scoring only runs if training succeeds.
- `environments` replaces the `%pip install` cells. Dependencies are declared once, version-controlled, and run on **serverless** job compute.
- The schedule is a weekly **safety net**. The *primary* trigger is drift (Module 08, Cell 8).

---

## 🛠 Hands-on

### Step 1 — Look at the script before deploying it

Open `bundle/src/train.py` in your Git folder. It's Modules 04 and 07 condensed: load `customer_features`, fit the pipeline, log and register it, set `@challenger`, then compare against `@champion` on the test split and promote if it's at least as good. Everything environment-specific arrives as an **argument**:

```py
p.add_argument("--catalog", required=True)
p.add_argument("--schema", required=True)
...
model_name = f"{args.catalog}.{args.schema}.{args.model_name}"
```

**What to point out**
- No hard-coded catalog, schema, or path anywhere. This is what makes a script *deployable*.
- It's a plain `.py` file, so it's easy to code-review and unit-test. Notebooks work as job tasks too (`notebook_task`), with parameters passed through widgets.

### Step 2 — Deploy to your dev target

**Option A: from the workspace (no install needed)**

1. In your Git folder, open `bundle/databricks.yml`.
2. Click the **Deployments** (🚀 bundle) icon in the editor sidebar.
3. Select target **dev** → **Deploy**.
4. When it finishes, the panel lists the deployed job **`[dev your_name] churn-training-dev`**. Click **Run**.

**Option B: from your laptop with the Databricks CLI**

```bash
cd bundle
databricks bundle validate -t dev          # checks YAML, resolves variables
databricks bundle deploy   -t dev          # creates/updates the job in the workspace
databricks bundle run      -t dev churn_training   # runs it and streams the status
```

**What to point out**
- `validate` catches typos and missing variables *before* anything changes.
- Run `deploy` twice: the second time nothing changes. Deploys are **idempotent**, and the files are the truth.
- Open **Jobs & Pipelines**: your job has the `[dev …]` prefix, a paused schedule, and tags `managed_by: asset-bundle`. Editing it in the UI shows a warning, because the source of truth is Git.

### Step 3 — Watch it run, then find it from a notebook

Create a notebook `09_notebook_to_production`.

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
me = w.current_user.me().user_name

job = next(j for j in w.jobs.list()                      # dev mode renamed it "[dev you] churn-training-dev"
           if j.settings.name.endswith("churn-training-dev") and j.creator_user_name == me)
print(f"Job ID: {job.job_id}  ← use this as RETRAIN_JOB_ID in Module 08")
for run in w.jobs.list_runs(job_id=job.job_id, limit=3):
    print(run.run_id, run.state.life_cycle_state, run.state.result_state)
```

**What to point out**
- Jobs are API objects. This is how Module 08's drift check triggers a retrain: `w.jobs.run_now(job_id=…)`.
- Open the run in the UI. Each task shows its logs, duration, and the MLflow run it produced. In the model page, a new version appears with a `promotion_note` tag written by the job.

### Step 4 — Promotion to staging and prod (instructor walkthrough)

Promotion is the **same command with a different target**, normally run by CI/CD, not by people:

```bash
databricks bundle deploy -t staging     # after the pull request is merged to main
databricks bundle deploy -t prod        # after staging checks pass, often with a manual approval
```

A minimal GitHub Actions workflow:

```yaml
# .github/workflows/deploy.yml
name: deploy-churn-bundle
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production                 # GitHub environment with required reviewers
    env:
      DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
      DATABRICKS_CLIENT_ID: ${{ secrets.SP_CLIENT_ID }}          # service principal (OAuth M2M)
      DATABRICKS_CLIENT_SECRET: ${{ secrets.SP_CLIENT_SECRET }}
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - run: databricks bundle validate -t staging && databricks bundle deploy -t staging
        working-directory: bundle
      - run: databricks bundle deploy -t prod
        working-directory: bundle
```

**What to point out**
- Production is deployed by a **service principal**, a robot identity, so no person's credentials run production jobs, and access is revoked by disabling one identity.
- The pull request *is* the change record: who changed what, who approved it, and when it went live.

---

## ✍️ Your turn (5 min)

The business wants scores **twice a week** (Monday and Thursday). Change the schedule in `resources/churn_job.yml` and redeploy to dev. What changed in the workspace?

<details>
<summary>Solution</summary>

```yaml
      schedule:
        quartz_cron_expression: "0 0 6 ? * MON,THU"
        timezone_id: Asia/Jakarta
```

Redeploy (UI **Deploy**, or `databricks bundle deploy -t dev`). The job's schedule updates in place and stays paused, because dev mode pauses schedules. In a real team this would be a one-line pull request, reviewed and then promoted to prod by CI/CD.

</details>

---

## ⚖️ Legacy vs Databricks

| | Legacy | Databricks Asset Bundles |
|---|---|---|
| Moving to prod | Export/import notebooks, find-and-replace names | `bundle deploy -t prod`, same files, different variables |
| Environment differences | Undocumented manual tweaks | Explicit per-target variables in Git |
| Job definitions | Clicked together in a UI | Declared in YAML, reviewed in pull requests |
| Who deployed what | Unknown | Git history + CI/CD logs; prod runs as a service principal |
| Parallel developers | Overwrite each other's jobs | `mode: development` prefixes and isolates everything |

---

**Check your understanding**

> *Twenty attendees deploy the same bundle to the same workspace. Why don't their jobs overwrite each other?*
>
> The `dev` target uses `mode: development`, which prefixes every resource with `[dev <username>]` and points the `schema` variable at each person's own schema. Each attendee gets their own job, writing to their own tables and model.

> *What's the difference between `train.py` running in dev and in prod?*
>
> Only the arguments: catalog, schema, and experiment come from the target's variables. The code, dependencies, and job structure are identical. That's the guarantee that what you tested is what runs in production.

---

[← Module 08 — Keep It Healthy](08-keep-it-healthy.md) | [Next: Module 10 — Wrap-Up →](10-wrap-up.md)
