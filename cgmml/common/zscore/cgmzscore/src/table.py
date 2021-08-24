import json
from pathlib import Path


class Table:
    def __init__(self, table_file: Path):
        self.table_file = table_file

    def load_table(self) -> None:
        table_file = open(self.table_file, 'r')
        self.table_value = json.load(table_file)

    def add_value(self) -> None:
        if 'Length' in self.table_value[0]:
            field_name = 'Length'
        elif 'Height' in self.table_value[0]:
            field_name = 'Height'
        elif 'Day' in self.table_value[0]:
            field_name = 'Day'
        else:
            raise Exception('error loading: %s' % self.table_value)
        self.field_name_dict = {'field_name': field_name}

    def append_value(self) -> dict:
        for value in self.table_value:
            self.field_name_dict.update({value[self.field_name_dict['field_name']]: value})
        return self.field_name_dict
