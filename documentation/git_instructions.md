# Basic GitHub Commands

This guide provides a quick reference for essential Git and GitHub commands.

| Description             | Command                       | Example                                      |
|-------------------------|-------------------------------|----------------------------------------------|
| **Create a new branch** | `git checkout -b <branch-name>` | `git checkout -b feature/user-authentication` |
| **Switch to a branch** | `git checkout <branch-name>`    | `git checkout main`                            |
| **Stage files for commit**| `git add <file-name(s)>`      | `git add script.js styles.css` or `git add .` |
| **Create a commit** | `git commit -m "<commit message>"` | `git commit -m "Implement user authentication"`|
| **Push local commits** | `git push <remote> <branch>`  | `git push origin main`                         |

## Command Details

### Create a New Branch
Use this command to create a new branch based on your current branch. It's best practice to create separate branches for new features or bug fixes to isolate your work.

* **Feature branches:** Start the branch name with `feature/` (e.g., `feature/user-login`).
* **Bug fix branches:** Start the branch name with `bugfix/` or `fix/` (e.g., `bugfix/bug_ticket_5`).

### Switch to a Branch
This command allows you to navigate between different branches in your local repository. Any changes you make will then be on the currently checked-out branch.

### Stage Files for Commit
Before you can save your changes with a commit, you need to tell Git which specific changes you want to include. The `git add` command stages these changes.

* `git add <file-name>`: Stages a specific file.
* `git add .`: Stages all unstaged changes in the current directory and its subdirectories. Be cautious with this to avoid accidentally staging unwanted changes.

### Create a Commit
A commit is a snapshot of your staged changes at a specific point in time. You should write clear and concise commit messages explaining what changes you've made.

* The `-m` flag allows you to include the commit message directly in the command.
* Keep commit messages informative. Good commit messages help you and your collaborators understand the history of the project.

**Important:** You need to use `git add` to stage your changes *before* creating a commit.

### Undo Current Changes On Local Branch
Discard all changes. Even commits
* git reset --hard HEAD
Undo commits
* git reset --soft HEAD^<commit count>

### Undo A Commit On Origin Branch
Copy the commit hash of the commit you want to revert the branch to
git reset --hard <commit-hash>
git push --force origin <branch-name>

### Push Local Commits
This command uploads your local commits to a remote repository (like GitHub).

* `<remote>`: This usually defaults to `origin`, which typically points to your main remote repository on GitHub.
* `<branch>`: Specifies the branch you want to push to on the remote repository.

**First-time push of a new local branch to a remote:**
```bash
git push -u origin <your-new-branch-name>

