from clearml import Task
from clearml.automation import HyperParameterOptimizer, GridSearch
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, LogUniformParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna

if __name__ == "__main__":
    task = Task.init(
        project_name='Masterarbeit/simple_snn',
        task_name='Gridsearch Task',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    task_id = Task.get_task(project_name="Masterarbeit", task_name="simple snn").id

    optimizer = HyperParameterOptimizer(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        base_task_id=task_id,
        # setting the hyperparameters to optimize
        hyper_parameters=[
            DiscreteParameterRange('num_steps', values=[2, 4, 8, 16, 32, 64, 128, 256]),
            #UniformIntegerParameterRange('batch_size', min_value=128, max_value=128, step_size=1),
            UniformParameterRange('beta', min_value=0.4, max_value=1, step_size=0.1),
            UniformParameterRange('gain', min_value=0.1, max_value=0.8, step_size=0.1)],
        # setting the objective metric we want to maximize/minimize
        objective_metric_title='validation metrics',
        objective_metric_series='loss',
        objective_metric_sign='min',

        # setting optimizer
        optimizer_class=GridSearch,

        # configuring optimization parameters
        execution_queue='default',
        max_number_of_concurrent_tasks=20,
        optimization_time_limit=60.,
        compute_time_limit=120,
        total_max_jobs=100,
        min_iteration_per_job=1000,
        max_iteration_per_job=10000,
    )

    optimizer.start()
