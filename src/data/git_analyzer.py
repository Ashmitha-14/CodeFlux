
import datetime
try:
    from git import Repo
except ImportError:
    Repo = None

class GitAnalyzer:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.repo = None
        if Repo:
            try:
                self.repo = Repo(repo_path)
            except Exception:
                pass # Handle non-git dirs gracefully

    def get_file_stats(self, file_path):
        if not self.repo:
            return {"commits": 0, "authors": 0, "churn": 0}
        
        try:
            commits = list(self.repo.iter_commits(paths=file_path))
            authors = set(c.author.email for c in commits)
            
            # Simple churn (number of commits)
            churn = len(commits)
            
            # Risk heuristic: Frequent fixes?
            fix_count = sum(1 for c in commits if "fix" in c.message.lower() or "bug" in c.message.lower())

            return {
                "commits": len(commits),
                "authors": len(authors),
                "churn": churn,
                "fix_count": fix_count,
                "last_modified": commits[0].committed_datetime.isoformat() if commits else None
            }
        except Exception as e:
            return {"error": str(e)}

