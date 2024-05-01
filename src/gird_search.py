from clearml import Task
from clearml.automation import HyperParameterOptimizer, GridSearch, Objective
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, LogUniformParameterRange, \
    DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna

if __name__ == "__main__":
    task = Task.init(
        project_name='Masterarbeit/simple_snn',
        task_name='Gridsearch fix gain=0.8',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    task_id = Task.get_task(project_name="Masterarbeit", task_name="simple snn").id

    gridSearch = GridSearch(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        base_task_id=task_id,
        # setting the hyperparameters to optimize
        hyper_parameters=[
            DiscreteParameterRange('num_steps', values=[2, 4, 8, 16, 32, 64, 128, 256]),
            # UniformIntegerParameterRange('batch_size', min_value=128, max_value=128, step_size=1),
            UniformParameterRange('beta', min_value=0.6, max_value=1, step_size=0.1),
            UniformParameterRange('gain', min_value=0.8, max_value=0.8, step_size=0.1)],
        # setting the objective metric we want to maximize/minimize
        objective_metric=Objective('validation metrics', 'loss'),
        num_concurrent_workers=10,

        # configuring optimization parameters
        execution_queue='default',
        compute_time_limit=12000,
        total_max_jobs=500,
    )

    gridSearch.start()
