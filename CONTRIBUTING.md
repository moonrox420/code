# Safe Procedure: Adding Local Files Without Overwriting Recent Fixes

> **Why this matters:** If `C:\Users\droxa\code` (or any local working directory) contains
> a `.git` folder pointed at the wrong remote, or if you blindly delete and re-clone on top
> of unsaved local changes, you risk losing recent fixes. Follow the steps below in order.

---

## 1. Verify you are in the correct repository

Open a terminal (PowerShell on Windows, bash on Linux/macOS) and run:

```powershell
# Windows (PowerShell)
cd C:\Users\droxa\code
git rev-parse --show-toplevel   # must print C:\Users\droxa\code
git remote -v                   # must show  moonrox420/code.git  for both fetch and push
```

```bash
# Linux / macOS
cd ~/code
git rev-parse --show-toplevel   # must print the correct absolute path
git remote -v                   # must show moonrox420/code.git
```

**If the remote is wrong** (e.g. it shows a different repo), stop here and follow
[Section 4 — Fixing a Wrong Remote](#4-fixing-a-wrong-remote) before continuing.

---

## 2. Check what files have changed locally

Before touching anything, get a clear picture of your local state:

```powershell
git status          # list modified, added, and untracked files
git diff            # show line-by-line changes to tracked files
git diff --cached   # show changes already staged with git add
```

Read this output carefully. Any file listed under *modified* or *untracked* that you
**do not** want to commit should not be staged.

---

## 3. Back up local files you want to keep

If you have local edits that are not yet committed (e.g. a README update, a config file,
a model path setting), copy them somewhere safe **before** any pull or re-clone:

```powershell
# Windows — copy the whole directory to a safe location
Copy-Item -Recurse C:\Users\droxa\code C:\Users\droxa\code_backup

# Or copy just the files you care about
Copy-Item C:\Users\droxa\code\README.md C:\Users\droxa\README.md.bak
```

```bash
# Linux / macOS
cp -r ~/code ~/code_backup
# or just the files you care about:
cp ~/code/README.md ~/README.md.bak
```

---

## 4. Fixing a Wrong Remote

If `git remote -v` shows the wrong repository URL, do **not** re-clone on top of the
existing folder — you could lose local changes. Instead:

```powershell
# 1. Back up local work first (Section 3 above).

# 2. Rename the accidental .git folder so it no longer controls the directory:
cd C:\Users\droxa
if (Test-Path .\.git) { Rename-Item .\.git .git.wrongrepo }

# 3. Clone the correct repo into a new folder:
git clone https://github.com/moonrox420/code.git C:\Users\droxa\code
cd C:\Users\droxa\code

# 4. Confirm:
git rev-parse --show-toplevel   # → C:\Users\droxa\code
git remote -v                   # → moonrox420/code.git
```

---

## 5. Restore only the changes you intend to commit

After you have a clean clone of `moonrox420/code`, **manually apply** only the edits you
want — do not copy entire directories back in one go, as that can overwrite upstream fixes.

**Example: restoring a README update**

```powershell
# Open your backup and the fresh clone side by side, then paste only the intended section.
# Or use a diff tool:
git diff README.md              # see what changed vs the upstream version
```

Stage and commit only the files you explicitly reviewed:

```powershell
git add README.md               # stage only this one file
git status                      # double-check nothing else is staged
git diff --cached               # review exactly what will be committed
git commit -m "docs: document GGUF local llama.cpp usage"
```

> **Tip:** Use `git add -p <file>` (patch mode) to stage individual hunks within a file
> rather than the entire file, giving you fine-grained control over what goes into the commit.

---

## 6. Avoid accidental overwrites when pulling

When you later run `git pull`, Git will refuse to overwrite uncommitted local changes.
To make pulls safe and predictable:

```powershell
# Option A — stash local work, pull, then reapply
git stash                       # save local changes temporarily
git pull origin main            # pull latest from the remote
git stash pop                   # reapply your local changes on top

# Option B — commit local work first, then pull (may require a merge)
git add README.md
git commit -m "wip: local edits before pull"
git pull origin main
```

Never use `git pull --force` or `git reset --hard origin/main` without first verifying
that you have no unsaved local work — both commands will **discard** uncommitted changes.

---

## 7. Push to `main` (or open a PR if `main` is protected)

```powershell
# Push directly (works if you have write access and main is not branch-protected)
git push origin main

# If main is protected, push a branch and open a Pull Request instead
git checkout -b docs/gguf-readme
git push -u origin docs/gguf-readme
# Then open a PR on GitHub: https://github.com/moonrox420/code/compare/docs/gguf-readme
```

---

## 8. Clean up

Once everything is committed and pushed, you can delete the backup and the renamed
wrong-repo folder:

```powershell
Remove-Item -Recurse -Force C:\Users\droxa\code_backup
Remove-Item -Recurse -Force C:\Users\droxa\.git.wrongrepo   # if created in Section 4
```

---

## Quick-reference checklist

- [ ] `git rev-parse --show-toplevel` shows the expected path
- [ ] `git remote -v` shows `moonrox420/code.git`
- [ ] Local files backed up before any pull or re-clone
- [ ] `git status` / `git diff` reviewed — no surprises
- [ ] Only intended files staged (`git add <file>`, not `git add .`)
- [ ] `git diff --cached` confirms exactly what will be committed
- [ ] Committed and pushed (or PR opened)
