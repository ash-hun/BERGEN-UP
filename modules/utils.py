import contextlib
from pandas import DataFrame

from rich import print
from rich.table import Table
from rich.errors import NotRenderableError

def rich_display_dataframe(df:DataFrame, title:str="Dataframe") -> None:
    """
    [ Display dataframe as table using rich library. ]

    * Args:
        df (pd.DataFrame): dataframe to display
        title (str, optional): title of the table. Defaults to "Dataframe".
    * Raises:
        NotRenderableError: if dataframe cannot be rendered
    * Returns:
        rich.table.Table: rich table
    """
    # Ensure dataframe contains only string values
    df = df.astype(str)

    table = Table(title=title)
    for col in df.columns:
        table.add_column(col)
    for row in df.values:
        with contextlib.suppress(NotRenderableError):
            table.add_row(*row)
    print(table)