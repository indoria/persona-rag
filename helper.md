## "git push -u origin main" hangs
```
GIT_TRACE=1 git push -u origin main
GIT_SSH_COMMAND="ssh -vvv" git push -u origin main


ssh -T git@github.com # (or git@gitlab.com, git@bitbucket.org, etc. based on your host)

- Ensure ssh-agent is running
eval "$(ssh-agent -s)"
ssh-agent -s
ssh-add ~/.ssh/id_rsa # or wherever your private key is located

git remote -v


git config --global --unset credential.helper
# Or, if you know which helper you're using (e.g., wincred, osxkeychain, manager)
# git credential-manager erase
# host=github.com (or gitlab.com etc.)

ssh -T git@github.com


chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
```

VS Code specific
```
In command palette (Cmd+Shift+P) search for "Reload Window"
```