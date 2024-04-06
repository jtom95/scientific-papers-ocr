import re

class LatexToKatexEquationTranslator:
    def __init__(self, text: str):
        self.text = text
        
    def parse(self):
        self._adjust_big()
        self._adjust_left_right()
        return self.text
        
    def _adjust_big(self):
        """ using \big is allowed, but not \big{<stuff>}"""
        
        keywords = ["big", "Big", "bigg", "Bigg"]
        for keyword in keywords:
            self.text = re.sub(rf"\\{keyword}{{(.*?)}}", rf"\\{keyword} \1", self.text)
    
    def _adjust_left_right(self):
        """
        \left and \right are used to automatically adjust the size of the brackets. However there must be 
        a matching pair of \left and \right. If there is not, then the parser will fail. This function
        will insert a matching pair of brackets if there is not one.
        The amount of left or right can be balanced by using \left. and \right. (note the period) at the beginning
        or end of the equation. 
        """
        
        left_matches = re.findall(r"\\left([(\[{|.\\])", self.text)
        right_matches = re.findall(r"\\right([)\]}|.\\])", self.text)

        
        if len(left_matches) > len(right_matches):
            self.text += "\\right." * (len(left_matches) - len(right_matches))
        if len(right_matches) > len(left_matches):
            self.text = "\\left." * (len(right_matches) - len(left_matches)) + self.text