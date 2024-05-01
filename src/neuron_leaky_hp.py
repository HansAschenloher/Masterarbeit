from clearml import Task
from clearml.automation import RandomSearch, Objective
from clearml.automation import UniformParameterRange, DiscreteParameterRange

if __name__ == "__main__":
    task = Task.init(
        project_name='Masterarbeit/LIF',
        task_name='GridSearch',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    task_id = Task.get_task(project_name="Masterarbeit/LIF", task_name="LIF base fashion_mnist").id

    randomSearch = RandomSearch(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        base_task_id=task_id,
        # setting the hyperparameters to optimize
        hyper_parameters=[
            DiscreteParameterRange('num_steps', values=[4, 8, 16, 32, 64, 128]),
            UniformParameterRange('beta', min_value=0.4, max_value=.99, step_size=0.05),
            DiscreteParameterRange('reset_mechanism', values=["zero", "subtract"]),
            DiscreteParameterRange('lr', values=[0.0001, 0.001, 0.01, 0.1]),
            DiscreteParameterRange('layer_config', values=[
                [28 * 28, 100, 10],
                [28 * 28, 10, 10, 10],
                [28 * 28, 20, 10, 10, 10],
            ]),
        ],
        # setting the objective metric we want to maximize/minimize
        objective_metric=Objective('validation metrics', 'loss'),
        num_concurrent_workers=1,

        # configuring optimization parameters
        execution_queue='default',
        compute_time_limit=12000,
        total_max_jobs=200,
        max_number_of_concurrent_tasks=5,
        min_iteration_per_job=1000,
        max_iteration_per_job=10000,
    )

    randomSearch.start()
