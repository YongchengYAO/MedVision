# Edits silently ignored because the install is a copy

## User Persona
A contributor modifying benchmark code who does not realise a plain install replaced their editable
install.

## Scenario Coverage
- Skill area: `environment-setup`
- Capability: install-mode diagnosis
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/installation.md`, `references/troubleshooting.md`;
  root `scripts/check_medvision_env.py`
- Trigger expectation: a silent-failure symptom with no error text, which requires the skill to know
  the trap rather than read a traceback.

## Expected Successful Behavior
Explains that a non-editable install places a copy in site-packages that shadows the checkout, so
edits appear to do nothing. Gives the one-line check that prints the resolved module file, and two
fixes: reinstall in editable mode, or run with the source directory on the module path. Warns that a
later plain install silently reverts editable mode, which is why repeated reinstalls did not help.

## Failure Signals
Suggesting a caching or bytecode issue; no way to verify which copy is imported; not mentioning that
a plain install undoes editable mode.
