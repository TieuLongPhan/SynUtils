from mod import ruleGMLString


def rule_isomorphism(
    rule_1: str, rule_2: str, morphism_type: str = "isomorphic"
) -> bool:
    """
    Evaluates if two GML-formatted rule representations are isomorphic or one is a
    subgraph of the other.

    Converts GML strings to `ruleGMLString` objects and uses these to check for:
    - 'isomorphic': Complete structural correspondence between both rules.
    - 'monomorphic': One rule being a subgraph of the other.

    Parameters:
    - rule_1 (str): GML string of the first rule.
    - rule_2 (str): GML string of the second rule.
    - morphism_type (str, optional): Type of morphism to check
    ('isomorphic' or 'monomorphic').

    Returns:
    - bool: True if the specified morphism condition is met, False otherwise.

    Raises:
    - Exception: Issues during GML parsing or morphism checking.
    """
    # Create ruleGMLString objects from the GML strings
    rule_obj_1 = ruleGMLString(rule_1)
    rule_obj_2 = ruleGMLString(rule_2)

    # Check the relationship based on morphism_type and return the result
    if morphism_type == "isomorphic":
        return rule_obj_1.isomorphism(rule_obj_2) == 1
    else:
        return rule_obj_1.monomorphism(rule_obj_2) == 1
