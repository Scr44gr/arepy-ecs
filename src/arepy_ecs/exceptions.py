class ECSException(Exception):
    """Base exception for the ECS package."""


class ComponentNotFoundError(ECSException):
    pass


class DuplicateComponentError(ECSException):
    pass


class EntityNotFoundError(ECSException):
    pass


class MaximumComponentsExceededError(ECSException):
    pass


class ResourceNotFoundError(ECSException):
    pass
