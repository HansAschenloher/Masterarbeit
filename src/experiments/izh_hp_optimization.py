from clearml import Task
from clearml.automation import HyperParameterOptimizer, GridSearch, RandomSearch
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, LogUniformParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna

if __name__ == "__main__":
    task = Task.init(
        project_name='Masterarbeit/izh_hp_opt',
        task_name='Automatic Hyper-Parameter Optimization',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    task_id = Task.get_task(project_name="Masterarbeit", task_name="IZH base task").id

    optimizer = HyperParameterOptimizer(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        base_task_id=task_id,
        # setting the hyperparameters to optimize
        hyper_parameters=[
            DiscreteParameterRange('neuron_type', values=["RS", "FS", "CH", "IB", "RZ"]),
            #DiscreteParameterRange('num_steps', values=[64]),
            DiscreteParameterRange('alpha', values=[0.5,0.6,0.7,0.8,0.9,0.95,0.99]),
            DiscreteParameterRange('beta', values=[0.4,0.5,0.6,0.7,0.8,0.9]),
            DiscreteParameterRange('lr', values=[0.1,0.01,0.001,0.0001]),
        ],
        # setting the objective metric we want to maximize/minimize
        objective_metric_title='validation metrics',
        objective_metric_series='loss',
        objective_metric_sign='min',

        # setting optimizer
        optimizer_class=RandomSearch,

        # configuring optimization parameters
        execution_queue='default',
        max_number_of_concurrent_tasks=2,
        optimization_time_limit=240.,
        compute_time_limit=120,
        total_max_jobs=200,
        min_iteration_per_job=1000,
        max_iteration_per_job=5000,
    )

    optimizer.start()
