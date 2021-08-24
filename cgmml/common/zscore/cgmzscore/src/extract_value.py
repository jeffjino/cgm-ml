from decimal import Decimal as D  # noqa: N817


class LmsBoxCox:
    def __init__(self, chart: str, weight: str, muac: str,
                 age_in_days: str, sex: str, height: str):
        self.chart = chart
        self.weight = weight
        self.muac = muac
        self.age_in_days = age_in_days
        self.sex = sex
        self.height = height

    def get_file_name_from_data(self):

        if self.chart == 'wfl' and D(self.height) > 110:
            table_chart = 'wfh'
            table_age = '2_5'
        elif self.chart == 'wfh' and D(self.height) < 65:
            table_chart = 'wfl'
            table_age = '0_2'
        else:
            table_chart = self.chart
            if self.chart == 'wfl':
                table_age = '0_2'
            if self.chart == 'wfh':
                table_age = '2_5'

        if self.sex == 'M':
            table_sex = 'boys'
        elif self.sex == 'F':
            table_sex = 'girls'

        if self.chart in ["wfa", "lhfa"]:
            table_age = "0_5"
            table_chart = self.chart

        self.table = "%(table_chart)s_%(table_sex)s_%(table_age)s" %\
            {"table_chart": table_chart,
             "table_sex": table_sex,
             "table_age": table_age}

    def get_lms_value(self, growth: dict):
        table_data = growth[self.table]
        if self.chart in ["wfh", "wfl"]:
            if D(self.height) < 45:
                raise Exception("too short")
            if D(self.height) > 120:
                raise TypeError("too tall")
            closest_height = float("{0:.1f}". format(D(self.height)))
            scores = table_data.get(str(closest_height))
            if scores is not None:
                self.scores = scores
            else:
                raise TypeError("Score not found for height :%s", (closest_height))

        if self.chart in ['wfa', 'lhfa']:
            scores = table_data.get(str(self.age_in_days))
            if scores is not None:
                self.scores = scores
            else:
                raise TypeError("Scores not found by Day: %s", self.age_in_days)

    def resolve_lms_value(self) -> int:
        skew = D(self.scores.get("L"))
        median = D(self.scores.get("M"))
        coff = D(self.scores.get("S"))
        if self.chart in ['wfa', 'wfl', 'wfh']:
            measurement = D(self.weight)
        elif self.chart == 'lhfa':
            measurement = D(self.height)
        else:
            raise NameError(f"Unknown chart name: {self.chart}")
        return skew, median, coff, measurement
