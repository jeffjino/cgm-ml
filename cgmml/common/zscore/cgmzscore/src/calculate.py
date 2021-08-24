"""https://www.who.int/childgrowth/standards/Chap_7.pdf"""
from decimal import Decimal as D  # noqa: N817


class Zscore:
    """Class to do calculation of z score"""

    def __init__(self, skew: int, median: int, coff: int, measurement: int):
        self.skew = skew
        self.median = median
        self.coff = coff
        self.measurement = measurement

    def calc_stdev(self, number: int):
        """This is not usual Standard Deviation please visit above PDF attached for clarification"""
        value = (1 + (self.skew * self.coff * number))**(1 / self.skew)
        stdev = self.median * value
        return stdev

    def z_score_measurement(self) -> float:
        """
         Z score
                  [y/M(t)]^L(t) - 1
           Zind =  -----------------
                      S(t)L(t)

                |       Zind            if |Zind| <= 3
                |
                |
                |       y - sd3pos
        Zind* = | 3 + ( ----------- )   if Zind > 3
                |         sd23pos
                |
                |
                |
                |        y - sd3neg
                | -3 + ( ----------- )  if Zind < -3
                |          sd23neg
        """

        numerator = (self.measurement / self.median)**self.skew - D(1.0)
        denominator = self.skew * self.coff
        z_score = numerator / denominator

        if D(z_score) > D(3):
            sd2pos = self.calc_stdev(2)
            sd3pos = self.calc_stdev(3)

            sd23pos = sd3pos - sd2pos

            z_score = 3 + ((self.measurement - sd3pos) / sd23pos)

            z_score = float(z_score.quantize(D('0.01')))

        elif D(z_score) < -3:
            sd2neg = self.calc_stdev(-2)
            sd3neg = self.calc_stdev(-3)

            sd23neg = sd2neg - sd3neg

            z_score = -3 + ((self.measurement - sd3neg) / sd23neg)
            z_score = float(z_score.quantize(D('0.01')))

        else:
            z_score = float(z_score.quantize(D('0.01')))

        return z_score
