from clearml import Task
from clearml.automation import DiscreteParameterRange
from clearml.automation import GridSearch, Objective

if __name__ == "__main__":
    task = Task.init(
        project_name='Masterarbeit/izh_hp_without_psp',
        task_name='Automatic Hyper-Parameter Optimization',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    task_id = Task.get_task(project_name="Masterarbeit/izh_hp_without_psp",
                            task_name="IZH base task without psp fashion").id

    randomSearch = GridSearch(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        base_task_id=task_id,
        # setting the hyperparameters to optimize
        hyper_parameters=[
            DiscreteParameterRange('neuron_type', values=["RS", "FS", "CH", "IB", "RZ"]),
            DiscreteParameterRange('lr', values=[0.1, 0.01, 0.001, 0.0001, 0.00001]),
            # DiscreteParameterRange('num_steps', values=[64]),
            # DiscreteParameterRange('alpha', values=[0.5,0.6,0.7,0.8,0.9,0.95,0.99]),
            # DiscreteParameterRange('beta', values=[0.4,0.5,0.6,0.7,0.8,0.9]),
        ],
        # setting the objective metric we want to maximize/minimize

        objective_metric=Objective('validation metrics', 'loss'),
        num_concurrent_workers=1,

        # configuring optimization parameters
        execution_queue='default',
        compute_time_limit=12000,
        total_max_jobs=3,
        max_number_of_concurrent_tasks=2,
        min_iteration_per_job=1000,
        max_iteration_per_job=5000,
    )

    randomSearch.start()
