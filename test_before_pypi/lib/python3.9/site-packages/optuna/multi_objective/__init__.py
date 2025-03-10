# TODO(nabenabe0928): Come up with any ways to remove this file.
# NOTE(nabenabe0928): Discuss when to remove this class.
migration_url = "https://github.com/optuna/optuna/discussions/5573"
raise ModuleNotFoundError(
    "\nThe features in `optuna.multi_objective` were integrated with the"
    "\nsingle objective optimization API and `optuna.multi_objective` were"
    "\ndeleted at v4.0.0. Please update your code based on the migration guide"
    f"\nat {migration_url}"
    "\nor downgrade your Optuna version."
)
