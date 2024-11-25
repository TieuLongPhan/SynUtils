from mod import ruleGMLString


def rule_isomorphism(rule_1: str, rule_2: str) -> bool:
    """
    Determines if two rule representations, given in GML format, are isomorphic.

    This function converts two GML strings into `ruleGMLString` objects and checks
    if these two objects are isomorphic. Isomorphism here is determined by the method
    `isomorphism` of the `ruleGMLString` class, which should return `1` for isomorphic
    structures and `0` otherwise.

    Parameters:
    - rule_1 (str): The GML string representation of the first rule.
    - rule_2 (str): The GML string representation of the second rule.

    Returns:
    - bool: `True` if the two rules are isomorphic; `False` otherwise.

    Raises:
    - Any exceptions thrown by the `ruleGMLString` initialization or methods should
      be documented here, if there are any known potential issues.
    """
    # Create ruleGMLString objects from the GML strings
    rule_obj_1 = ruleGMLString(rule_1)
    rule_obj_2 = ruleGMLString(rule_2)

    # Check for isomorphism and return the result
    return rule_obj_1.isomorphism(rule_obj_2) == 1
