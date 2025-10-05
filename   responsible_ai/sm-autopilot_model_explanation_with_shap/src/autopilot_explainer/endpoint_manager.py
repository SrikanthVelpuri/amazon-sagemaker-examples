"""Managed SageMaker Endpoint context manager."""

import logging
import boto3
from typing import Optional

logger = logging.getLogger(__name__)


class ManagedEndpoint:
    """Context manager for SageMaker endpoints with optional auto-deletion.

    Args:
        ep_name: Endpoint name
        auto_delete: If True, delete endpoint on exit (default: False)
        region_name: AWS region (default: uses boto3 session region)
    """

    def __init__(
        self, ep_name: str, auto_delete: bool = False, region_name: Optional[str] = None
    ):
        self.name = ep_name
        self.auto_delete = auto_delete
        self.in_service = False
        self.region = region_name or boto3.Session().region_name
        self.sm = boto3.Session().client(service_name="sagemaker", region_name=self.region)
        logger.info(f"Initialized ManagedEndpoint for: {ep_name}")

    def __enter__(self):
        """Verify endpoint is in service."""
        try:
            endpoint_description = self.sm.describe_endpoint(EndpointName=self.name)
            status = endpoint_description["EndpointStatus"]

            if status == "InService":
                self.in_service = True
                logger.info(f"Endpoint {self.name} is InService")
            else:
                logger.warning(f"Endpoint {self.name} status: {status}")
                raise RuntimeError(f"Endpoint {self.name} is not InService (status: {status})")

            return self
        except Exception as e:
            logger.error(f"Error entering endpoint context: {str(e)}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Delete endpoint if auto_delete is enabled."""
        if self.in_service and self.auto_delete:
            try:
                logger.info(f"Deleting endpoint: {self.name}")
                self.sm.delete_endpoint(EndpointName=self.name)
                self.sm.get_waiter("endpoint_deleted").wait(EndpointName=self.name)
                self.in_service = False
                logger.info(f"Successfully deleted endpoint: {self.name}")
            except Exception as e:
                logger.error(f"Error deleting endpoint: {str(e)}")
                raise
