import click


from .predict import predict
from .train import train
from .transform import transform


@click.group()
def cli():

    """
    You can use following commands:
    - predict
    - train
    - transform

    """
    pass


cli.add_command(predict)
cli.add_command(train)
cli.add_command(transform)
