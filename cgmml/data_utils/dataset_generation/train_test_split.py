import argparse
import random

import dbutils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_connection_file', default="dbconnection.json",
                        help='json file with secrets to connect to database')

    args = parser.parse_args()
    db_connection_file = args.db_connection_file

    connector = dbutils.connect_to_main_database(db_connection_file)

    select_uncategorized_persons = """
    SELECT id from person p
    WHERE NOT EXISTS
    (SELECT person_id FROM child_data_category
    WHERE person_id = p.id);
    """

    person_ids = connector.execute(select_uncategorized_persons, fetch_all=True)
    person_ids = [person_id[0] for person_id in person_ids]

    for person_id in person_ids:
        # data category id {1: Train, 2: Test}
        data_category_id = random.choices([1, 2], weights=(75, 25), k=1)[0]
        insert_statement = f"""INSERT INTO child_data_category
                                (person_id, data_category_id)
                                VALUES ('{person_id}', {data_category_id});"""
        connector.execute(insert_statement)


if __name__ == "__main__":
    main()
