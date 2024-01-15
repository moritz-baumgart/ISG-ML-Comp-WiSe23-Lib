def get_categorical_labels(with_ft_2=False):
    ft_no = [10, 12, 20]
    if with_ft_2:
        ft_no.insert(0, 2)
    return [f"feature_{x}" for x in ft_no]
