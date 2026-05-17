"""Generic design patterns for the HHO kernel."""

def resettable_lazy_property(func):
    """
    Decorator that turns a method into a resettable, lazy-evaluated property.
    The result is stored in self._<property_name>.
    """
    attr_name = f"_{func.__name__}"

    @property
    def wrapper(self):
        if getattr(self, attr_name) is None:
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    wrapper.__doc__ = func.__doc__
    return wrapper
