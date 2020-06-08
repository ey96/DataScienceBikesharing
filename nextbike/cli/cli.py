import click

from .predict import predict
from .train import train
from .transform import transform


@click.group()
def cli():
    """
    Mock method.
    Only to register the different commands (i.e., predict, train, transform)
    We don't specify a specific order in which you need to run these commands.
    It's totally up to you.

    """
    pass


# cli.add_command(predict)
cli.add_command(train)
cli.add_command(transform)
