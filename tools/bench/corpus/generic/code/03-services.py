"""Service layer: validation, not-found handling and logging around the repositories."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NotFoundError(Exception):
    """Raised when a record does not exist."""


class ValidationError(Exception):
    """Raised when a record is not valid."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class UserService:
    """Business rules for users."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_user(self, data: dict) -> dict:
        """Validate and store a new user."""
        item = User.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid user: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created user %s", item.id)
        return item.to_dict()

    def get_user(self, item_id: int) -> dict:
        """Return one user or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"user {item_id} not found")
        return item.to_dict()

    def list_users(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of users with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_user(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing user."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"user {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated user %s", item_id)
        return item.to_dict()

    def delete_user(self, item_id: int) -> None:
        """Delete a user or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"user {item_id} not found")
        logger.info("deleted user %s", item_id)


class ProductService:
    """Business rules for products."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_product(self, data: dict) -> dict:
        """Validate and store a new product."""
        item = Product.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid product: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created product %s", item.id)
        return item.to_dict()

    def get_product(self, item_id: int) -> dict:
        """Return one product or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"product {item_id} not found")
        return item.to_dict()

    def list_products(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of products with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_product(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing product."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"product {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated product %s", item_id)
        return item.to_dict()

    def delete_product(self, item_id: int) -> None:
        """Delete a product or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"product {item_id} not found")
        logger.info("deleted product %s", item_id)


class OrderService:
    """Business rules for orders."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_order(self, data: dict) -> dict:
        """Validate and store a new order."""
        item = Order.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid order: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created order %s", item.id)
        return item.to_dict()

    def get_order(self, item_id: int) -> dict:
        """Return one order or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"order {item_id} not found")
        return item.to_dict()

    def list_orders(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of orders with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_order(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing order."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"order {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated order %s", item_id)
        return item.to_dict()

    def delete_order(self, item_id: int) -> None:
        """Delete a order or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"order {item_id} not found")
        logger.info("deleted order %s", item_id)


class OrderItemService:
    """Business rules for order items."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_order_item(self, data: dict) -> dict:
        """Validate and store a new order item."""
        item = OrderItem.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid order item: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created order item %s", item.id)
        return item.to_dict()

    def get_order_item(self, item_id: int) -> dict:
        """Return one order item or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"order item {item_id} not found")
        return item.to_dict()

    def list_order_items(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of order items with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_order_item(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing order item."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"order item {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated order item %s", item_id)
        return item.to_dict()

    def delete_order_item(self, item_id: int) -> None:
        """Delete a order item or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"order item {item_id} not found")
        logger.info("deleted order item %s", item_id)


class CustomerService:
    """Business rules for customers."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_customer(self, data: dict) -> dict:
        """Validate and store a new customer."""
        item = Customer.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid customer: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created customer %s", item.id)
        return item.to_dict()

    def get_customer(self, item_id: int) -> dict:
        """Return one customer or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"customer {item_id} not found")
        return item.to_dict()

    def list_customers(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of customers with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_customer(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing customer."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"customer {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated customer %s", item_id)
        return item.to_dict()

    def delete_customer(self, item_id: int) -> None:
        """Delete a customer or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"customer {item_id} not found")
        logger.info("deleted customer %s", item_id)


class InvoiceService:
    """Business rules for invoices."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_invoice(self, data: dict) -> dict:
        """Validate and store a new invoice."""
        item = Invoice.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid invoice: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created invoice %s", item.id)
        return item.to_dict()

    def get_invoice(self, item_id: int) -> dict:
        """Return one invoice or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"invoice {item_id} not found")
        return item.to_dict()

    def list_invoices(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of invoices with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_invoice(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing invoice."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"invoice {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated invoice %s", item_id)
        return item.to_dict()

    def delete_invoice(self, item_id: int) -> None:
        """Delete a invoice or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"invoice {item_id} not found")
        logger.info("deleted invoice %s", item_id)


class PaymentService:
    """Business rules for payments."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_payment(self, data: dict) -> dict:
        """Validate and store a new payment."""
        item = Payment.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid payment: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created payment %s", item.id)
        return item.to_dict()

    def get_payment(self, item_id: int) -> dict:
        """Return one payment or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"payment {item_id} not found")
        return item.to_dict()

    def list_payments(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of payments with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_payment(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing payment."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"payment {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated payment %s", item_id)
        return item.to_dict()

    def delete_payment(self, item_id: int) -> None:
        """Delete a payment or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"payment {item_id} not found")
        logger.info("deleted payment %s", item_id)


class CategoryService:
    """Business rules for categories."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_category(self, data: dict) -> dict:
        """Validate and store a new category."""
        item = Category.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid category: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created category %s", item.id)
        return item.to_dict()

    def get_category(self, item_id: int) -> dict:
        """Return one category or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"category {item_id} not found")
        return item.to_dict()

    def list_categories(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of categories with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_category(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing category."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"category {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated category %s", item_id)
        return item.to_dict()

    def delete_category(self, item_id: int) -> None:
        """Delete a category or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"category {item_id} not found")
        logger.info("deleted category %s", item_id)


class ReviewService:
    """Business rules for reviews."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_review(self, data: dict) -> dict:
        """Validate and store a new review."""
        item = Review.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid review: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created review %s", item.id)
        return item.to_dict()

    def get_review(self, item_id: int) -> dict:
        """Return one review or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"review {item_id} not found")
        return item.to_dict()

    def list_reviews(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of reviews with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_review(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing review."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"review {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated review %s", item_id)
        return item.to_dict()

    def delete_review(self, item_id: int) -> None:
        """Delete a review or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"review {item_id} not found")
        logger.info("deleted review %s", item_id)


class AddressService:
    """Business rules for addresses."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_address(self, data: dict) -> dict:
        """Validate and store a new address."""
        item = Address.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid address: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created address %s", item.id)
        return item.to_dict()

    def get_address(self, item_id: int) -> dict:
        """Return one address or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"address {item_id} not found")
        return item.to_dict()

    def list_addresses(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of addresses with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_address(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing address."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"address {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated address %s", item_id)
        return item.to_dict()

    def delete_address(self, item_id: int) -> None:
        """Delete a address or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"address {item_id} not found")
        logger.info("deleted address %s", item_id)


class ShipmentService:
    """Business rules for shipments."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_shipment(self, data: dict) -> dict:
        """Validate and store a new shipment."""
        item = Shipment.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid shipment: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created shipment %s", item.id)
        return item.to_dict()

    def get_shipment(self, item_id: int) -> dict:
        """Return one shipment or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"shipment {item_id} not found")
        return item.to_dict()

    def list_shipments(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of shipments with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_shipment(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing shipment."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"shipment {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated shipment %s", item_id)
        return item.to_dict()

    def delete_shipment(self, item_id: int) -> None:
        """Delete a shipment or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"shipment {item_id} not found")
        logger.info("deleted shipment %s", item_id)


class WarehouseService:
    """Business rules for warehouses."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_warehouse(self, data: dict) -> dict:
        """Validate and store a new warehouse."""
        item = Warehouse.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid warehouse: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created warehouse %s", item.id)
        return item.to_dict()

    def get_warehouse(self, item_id: int) -> dict:
        """Return one warehouse or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"warehouse {item_id} not found")
        return item.to_dict()

    def list_warehouses(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of warehouses with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_warehouse(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing warehouse."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"warehouse {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated warehouse %s", item_id)
        return item.to_dict()

    def delete_warehouse(self, item_id: int) -> None:
        """Delete a warehouse or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"warehouse {item_id} not found")
        logger.info("deleted warehouse %s", item_id)


class SupplierService:
    """Business rules for suppliers."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_supplier(self, data: dict) -> dict:
        """Validate and store a new supplier."""
        item = Supplier.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid supplier: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created supplier %s", item.id)
        return item.to_dict()

    def get_supplier(self, item_id: int) -> dict:
        """Return one supplier or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"supplier {item_id} not found")
        return item.to_dict()

    def list_suppliers(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of suppliers with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_supplier(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing supplier."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"supplier {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated supplier %s", item_id)
        return item.to_dict()

    def delete_supplier(self, item_id: int) -> None:
        """Delete a supplier or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"supplier {item_id} not found")
        logger.info("deleted supplier %s", item_id)


class CouponService:
    """Business rules for coupons."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_coupon(self, data: dict) -> dict:
        """Validate and store a new coupon."""
        item = Coupon.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid coupon: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created coupon %s", item.id)
        return item.to_dict()

    def get_coupon(self, item_id: int) -> dict:
        """Return one coupon or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"coupon {item_id} not found")
        return item.to_dict()

    def list_coupons(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of coupons with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_coupon(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing coupon."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"coupon {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated coupon %s", item_id)
        return item.to_dict()

    def delete_coupon(self, item_id: int) -> None:
        """Delete a coupon or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"coupon {item_id} not found")
        logger.info("deleted coupon %s", item_id)


class EmployeeService:
    """Business rules for employees."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_employee(self, data: dict) -> dict:
        """Validate and store a new employee."""
        item = Employee.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid employee: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created employee %s", item.id)
        return item.to_dict()

    def get_employee(self, item_id: int) -> dict:
        """Return one employee or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"employee {item_id} not found")
        return item.to_dict()

    def list_employees(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of employees with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_employee(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing employee."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"employee {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated employee %s", item_id)
        return item.to_dict()

    def delete_employee(self, item_id: int) -> None:
        """Delete a employee or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"employee {item_id} not found")
        logger.info("deleted employee %s", item_id)


class DepartmentService:
    """Business rules for departments."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_department(self, data: dict) -> dict:
        """Validate and store a new department."""
        item = Department.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid department: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created department %s", item.id)
        return item.to_dict()

    def get_department(self, item_id: int) -> dict:
        """Return one department or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"department {item_id} not found")
        return item.to_dict()

    def list_departments(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of departments with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_department(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing department."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"department {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated department %s", item_id)
        return item.to_dict()

    def delete_department(self, item_id: int) -> None:
        """Delete a department or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"department {item_id} not found")
        logger.info("deleted department %s", item_id)


class ProjectService:
    """Business rules for projects."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_project(self, data: dict) -> dict:
        """Validate and store a new project."""
        item = Project.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid project: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created project %s", item.id)
        return item.to_dict()

    def get_project(self, item_id: int) -> dict:
        """Return one project or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"project {item_id} not found")
        return item.to_dict()

    def list_projects(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of projects with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_project(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing project."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"project {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated project %s", item_id)
        return item.to_dict()

    def delete_project(self, item_id: int) -> None:
        """Delete a project or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"project {item_id} not found")
        logger.info("deleted project %s", item_id)


class TaskService:
    """Business rules for tasks."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_task(self, data: dict) -> dict:
        """Validate and store a new task."""
        item = Task.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid task: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created task %s", item.id)
        return item.to_dict()

    def get_task(self, item_id: int) -> dict:
        """Return one task or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"task {item_id} not found")
        return item.to_dict()

    def list_tasks(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of tasks with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_task(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing task."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"task {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated task %s", item_id)
        return item.to_dict()

    def delete_task(self, item_id: int) -> None:
        """Delete a task or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"task {item_id} not found")
        logger.info("deleted task %s", item_id)


class CommentService:
    """Business rules for comments."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_comment(self, data: dict) -> dict:
        """Validate and store a new comment."""
        item = Comment.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid comment: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created comment %s", item.id)
        return item.to_dict()

    def get_comment(self, item_id: int) -> dict:
        """Return one comment or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"comment {item_id} not found")
        return item.to_dict()

    def list_comments(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of comments with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_comment(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing comment."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"comment {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated comment %s", item_id)
        return item.to_dict()

    def delete_comment(self, item_id: int) -> None:
        """Delete a comment or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"comment {item_id} not found")
        logger.info("deleted comment %s", item_id)


class TagService:
    """Business rules for tags."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_tag(self, data: dict) -> dict:
        """Validate and store a new tag."""
        item = Tag.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid tag: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created tag %s", item.id)
        return item.to_dict()

    def get_tag(self, item_id: int) -> dict:
        """Return one tag or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"tag {item_id} not found")
        return item.to_dict()

    def list_tags(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of tags with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_tag(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing tag."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"tag {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated tag %s", item_id)
        return item.to_dict()

    def delete_tag(self, item_id: int) -> None:
        """Delete a tag or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"tag {item_id} not found")
        logger.info("deleted tag %s", item_id)


class ArticleService:
    """Business rules for articles."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_article(self, data: dict) -> dict:
        """Validate and store a new article."""
        item = Article.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid article: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created article %s", item.id)
        return item.to_dict()

    def get_article(self, item_id: int) -> dict:
        """Return one article or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"article {item_id} not found")
        return item.to_dict()

    def list_articles(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of articles with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_article(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing article."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"article {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated article %s", item_id)
        return item.to_dict()

    def delete_article(self, item_id: int) -> None:
        """Delete a article or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"article {item_id} not found")
        logger.info("deleted article %s", item_id)


class AuthorService:
    """Business rules for authors."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_author(self, data: dict) -> dict:
        """Validate and store a new author."""
        item = Author.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid author: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created author %s", item.id)
        return item.to_dict()

    def get_author(self, item_id: int) -> dict:
        """Return one author or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"author {item_id} not found")
        return item.to_dict()

    def list_authors(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of authors with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_author(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing author."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"author {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated author %s", item_id)
        return item.to_dict()

    def delete_author(self, item_id: int) -> None:
        """Delete a author or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"author {item_id} not found")
        logger.info("deleted author %s", item_id)


class BookService:
    """Business rules for books."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_book(self, data: dict) -> dict:
        """Validate and store a new book."""
        item = Book.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid book: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created book %s", item.id)
        return item.to_dict()

    def get_book(self, item_id: int) -> dict:
        """Return one book or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"book {item_id} not found")
        return item.to_dict()

    def list_books(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of books with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_book(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing book."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"book {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated book %s", item_id)
        return item.to_dict()

    def delete_book(self, item_id: int) -> None:
        """Delete a book or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"book {item_id} not found")
        logger.info("deleted book %s", item_id)


class LibraryService:
    """Business rules for libraries."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_library(self, data: dict) -> dict:
        """Validate and store a new library."""
        item = Library.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid library: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created library %s", item.id)
        return item.to_dict()

    def get_library(self, item_id: int) -> dict:
        """Return one library or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"library {item_id} not found")
        return item.to_dict()

    def list_libraries(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of libraries with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_library(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing library."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"library {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated library %s", item_id)
        return item.to_dict()

    def delete_library(self, item_id: int) -> None:
        """Delete a library or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"library {item_id} not found")
        logger.info("deleted library %s", item_id)


class LoanService:
    """Business rules for loans."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_loan(self, data: dict) -> dict:
        """Validate and store a new loan."""
        item = Loan.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid loan: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created loan %s", item.id)
        return item.to_dict()

    def get_loan(self, item_id: int) -> dict:
        """Return one loan or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"loan {item_id} not found")
        return item.to_dict()

    def list_loans(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of loans with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_loan(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing loan."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"loan {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated loan %s", item_id)
        return item.to_dict()

    def delete_loan(self, item_id: int) -> None:
        """Delete a loan or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"loan {item_id} not found")
        logger.info("deleted loan %s", item_id)


class MemberService:
    """Business rules for members."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_member(self, data: dict) -> dict:
        """Validate and store a new member."""
        item = Member.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid member: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created member %s", item.id)
        return item.to_dict()

    def get_member(self, item_id: int) -> dict:
        """Return one member or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"member {item_id} not found")
        return item.to_dict()

    def list_members(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of members with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_member(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing member."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"member {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated member %s", item_id)
        return item.to_dict()

    def delete_member(self, item_id: int) -> None:
        """Delete a member or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"member {item_id} not found")
        logger.info("deleted member %s", item_id)


class EventService:
    """Business rules for events."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_event(self, data: dict) -> dict:
        """Validate and store a new event."""
        item = Event.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid event: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created event %s", item.id)
        return item.to_dict()

    def get_event(self, item_id: int) -> dict:
        """Return one event or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"event {item_id} not found")
        return item.to_dict()

    def list_events(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of events with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_event(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing event."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"event {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated event %s", item_id)
        return item.to_dict()

    def delete_event(self, item_id: int) -> None:
        """Delete a event or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"event {item_id} not found")
        logger.info("deleted event %s", item_id)


class TicketService:
    """Business rules for tickets."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_ticket(self, data: dict) -> dict:
        """Validate and store a new ticket."""
        item = Ticket.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid ticket: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created ticket %s", item.id)
        return item.to_dict()

    def get_ticket(self, item_id: int) -> dict:
        """Return one ticket or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"ticket {item_id} not found")
        return item.to_dict()

    def list_tickets(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of tickets with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_ticket(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing ticket."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"ticket {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated ticket %s", item_id)
        return item.to_dict()

    def delete_ticket(self, item_id: int) -> None:
        """Delete a ticket or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"ticket {item_id} not found")
        logger.info("deleted ticket %s", item_id)


class VenueService:
    """Business rules for venues."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_venue(self, data: dict) -> dict:
        """Validate and store a new venue."""
        item = Venue.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid venue: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created venue %s", item.id)
        return item.to_dict()

    def get_venue(self, item_id: int) -> dict:
        """Return one venue or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"venue {item_id} not found")
        return item.to_dict()

    def list_venues(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of venues with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_venue(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing venue."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"venue {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated venue %s", item_id)
        return item.to_dict()

    def delete_venue(self, item_id: int) -> None:
        """Delete a venue or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"venue {item_id} not found")
        logger.info("deleted venue %s", item_id)


class VehicleService:
    """Business rules for vehicles."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_vehicle(self, data: dict) -> dict:
        """Validate and store a new vehicle."""
        item = Vehicle.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid vehicle: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created vehicle %s", item.id)
        return item.to_dict()

    def get_vehicle(self, item_id: int) -> dict:
        """Return one vehicle or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"vehicle {item_id} not found")
        return item.to_dict()

    def list_vehicles(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of vehicles with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_vehicle(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing vehicle."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"vehicle {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated vehicle %s", item_id)
        return item.to_dict()

    def delete_vehicle(self, item_id: int) -> None:
        """Delete a vehicle or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"vehicle {item_id} not found")
        logger.info("deleted vehicle %s", item_id)


class DriverService:
    """Business rules for drivers."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_driver(self, data: dict) -> dict:
        """Validate and store a new driver."""
        item = Driver.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid driver: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created driver %s", item.id)
        return item.to_dict()

    def get_driver(self, item_id: int) -> dict:
        """Return one driver or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"driver {item_id} not found")
        return item.to_dict()

    def list_drivers(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of drivers with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_driver(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing driver."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"driver {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated driver %s", item_id)
        return item.to_dict()

    def delete_driver(self, item_id: int) -> None:
        """Delete a driver or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"driver {item_id} not found")
        logger.info("deleted driver %s", item_id)


class TripService:
    """Business rules for trips."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_trip(self, data: dict) -> dict:
        """Validate and store a new trip."""
        item = Trip.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid trip: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created trip %s", item.id)
        return item.to_dict()

    def get_trip(self, item_id: int) -> dict:
        """Return one trip or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"trip {item_id} not found")
        return item.to_dict()

    def list_trips(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of trips with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_trip(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing trip."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"trip {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated trip %s", item_id)
        return item.to_dict()

    def delete_trip(self, item_id: int) -> None:
        """Delete a trip or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"trip {item_id} not found")
        logger.info("deleted trip %s", item_id)


class PatientService:
    """Business rules for patients."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_patient(self, data: dict) -> dict:
        """Validate and store a new patient."""
        item = Patient.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid patient: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created patient %s", item.id)
        return item.to_dict()

    def get_patient(self, item_id: int) -> dict:
        """Return one patient or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"patient {item_id} not found")
        return item.to_dict()

    def list_patients(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of patients with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_patient(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing patient."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"patient {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated patient %s", item_id)
        return item.to_dict()

    def delete_patient(self, item_id: int) -> None:
        """Delete a patient or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"patient {item_id} not found")
        logger.info("deleted patient %s", item_id)


class DoctorService:
    """Business rules for doctors."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_doctor(self, data: dict) -> dict:
        """Validate and store a new doctor."""
        item = Doctor.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid doctor: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created doctor %s", item.id)
        return item.to_dict()

    def get_doctor(self, item_id: int) -> dict:
        """Return one doctor or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"doctor {item_id} not found")
        return item.to_dict()

    def list_doctors(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of doctors with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_doctor(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing doctor."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"doctor {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated doctor %s", item_id)
        return item.to_dict()

    def delete_doctor(self, item_id: int) -> None:
        """Delete a doctor or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"doctor {item_id} not found")
        logger.info("deleted doctor %s", item_id)


class AppointmentService:
    """Business rules for appointments."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_appointment(self, data: dict) -> dict:
        """Validate and store a new appointment."""
        item = Appointment.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid appointment: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created appointment %s", item.id)
        return item.to_dict()

    def get_appointment(self, item_id: int) -> dict:
        """Return one appointment or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"appointment {item_id} not found")
        return item.to_dict()

    def list_appointments(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of appointments with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_appointment(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing appointment."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"appointment {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated appointment %s", item_id)
        return item.to_dict()

    def delete_appointment(self, item_id: int) -> None:
        """Delete a appointment or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"appointment {item_id} not found")
        logger.info("deleted appointment %s", item_id)


class CourseService:
    """Business rules for courses."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_course(self, data: dict) -> dict:
        """Validate and store a new course."""
        item = Course.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid course: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created course %s", item.id)
        return item.to_dict()

    def get_course(self, item_id: int) -> dict:
        """Return one course or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"course {item_id} not found")
        return item.to_dict()

    def list_courses(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of courses with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_course(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing course."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"course {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated course %s", item_id)
        return item.to_dict()

    def delete_course(self, item_id: int) -> None:
        """Delete a course or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"course {item_id} not found")
        logger.info("deleted course %s", item_id)


class StudentService:
    """Business rules for students."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_student(self, data: dict) -> dict:
        """Validate and store a new student."""
        item = Student.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid student: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created student %s", item.id)
        return item.to_dict()

    def get_student(self, item_id: int) -> dict:
        """Return one student or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"student {item_id} not found")
        return item.to_dict()

    def list_students(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of students with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_student(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing student."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"student {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated student %s", item_id)
        return item.to_dict()

    def delete_student(self, item_id: int) -> None:
        """Delete a student or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"student {item_id} not found")
        logger.info("deleted student %s", item_id)


class EnrollmentService:
    """Business rules for enrollments."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_enrollment(self, data: dict) -> dict:
        """Validate and store a new enrollment."""
        item = Enrollment.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid enrollment: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created enrollment %s", item.id)
        return item.to_dict()

    def get_enrollment(self, item_id: int) -> dict:
        """Return one enrollment or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"enrollment {item_id} not found")
        return item.to_dict()

    def list_enrollments(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of enrollments with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_enrollment(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing enrollment."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"enrollment {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated enrollment %s", item_id)
        return item.to_dict()

    def delete_enrollment(self, item_id: int) -> None:
        """Delete a enrollment or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"enrollment {item_id} not found")
        logger.info("deleted enrollment %s", item_id)


class RoomService:
    """Business rules for rooms."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_room(self, data: dict) -> dict:
        """Validate and store a new room."""
        item = Room.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid room: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created room %s", item.id)
        return item.to_dict()

    def get_room(self, item_id: int) -> dict:
        """Return one room or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"room {item_id} not found")
        return item.to_dict()

    def list_rooms(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of rooms with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_room(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing room."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"room {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated room %s", item_id)
        return item.to_dict()

    def delete_room(self, item_id: int) -> None:
        """Delete a room or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"room {item_id} not found")
        logger.info("deleted room %s", item_id)


class BookingService:
    """Business rules for bookings."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_booking(self, data: dict) -> dict:
        """Validate and store a new booking."""
        item = Booking.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid booking: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created booking %s", item.id)
        return item.to_dict()

    def get_booking(self, item_id: int) -> dict:
        """Return one booking or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"booking {item_id} not found")
        return item.to_dict()

    def list_bookings(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of bookings with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_booking(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing booking."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"booking {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated booking %s", item_id)
        return item.to_dict()

    def delete_booking(self, item_id: int) -> None:
        """Delete a booking or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"booking {item_id} not found")
        logger.info("deleted booking %s", item_id)


class SubscriptionService:
    """Business rules for subscriptions."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_subscription(self, data: dict) -> dict:
        """Validate and store a new subscription."""
        item = Subscription.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid subscription: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created subscription %s", item.id)
        return item.to_dict()

    def get_subscription(self, item_id: int) -> dict:
        """Return one subscription or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"subscription {item_id} not found")
        return item.to_dict()

    def list_subscriptions(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of subscriptions with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_subscription(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing subscription."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"subscription {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated subscription %s", item_id)
        return item.to_dict()

    def delete_subscription(self, item_id: int) -> None:
        """Delete a subscription or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"subscription {item_id} not found")
        logger.info("deleted subscription %s", item_id)


class NotificationService:
    """Business rules for notifications."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_notification(self, data: dict) -> dict:
        """Validate and store a new notification."""
        item = Notification.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid notification: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created notification %s", item.id)
        return item.to_dict()

    def get_notification(self, item_id: int) -> dict:
        """Return one notification or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"notification {item_id} not found")
        return item.to_dict()

    def list_notifications(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of notifications with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_notification(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing notification."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"notification {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated notification %s", item_id)
        return item.to_dict()

    def delete_notification(self, item_id: int) -> None:
        """Delete a notification or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"notification {item_id} not found")
        logger.info("deleted notification %s", item_id)


class SessionService:
    """Business rules for sessions."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_session(self, data: dict) -> dict:
        """Validate and store a new session."""
        item = Session.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid session: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created session %s", item.id)
        return item.to_dict()

    def get_session(self, item_id: int) -> dict:
        """Return one session or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"session {item_id} not found")
        return item.to_dict()

    def list_sessions(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of sessions with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_session(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing session."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"session {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated session %s", item_id)
        return item.to_dict()

    def delete_session(self, item_id: int) -> None:
        """Delete a session or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"session {item_id} not found")
        logger.info("deleted session %s", item_id)


class AuditLogService:
    """Business rules for audit logs."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_audit_log(self, data: dict) -> dict:
        """Validate and store a new audit log."""
        item = AuditLog.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid audit log: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created audit log %s", item.id)
        return item.to_dict()

    def get_audit_log(self, item_id: int) -> dict:
        """Return one audit log or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"audit log {item_id} not found")
        return item.to_dict()

    def list_audit_logs(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of audit logs with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_audit_log(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing audit log."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"audit log {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated audit log %s", item_id)
        return item.to_dict()

    def delete_audit_log(self, item_id: int) -> None:
        """Delete a audit log or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"audit log {item_id} not found")
        logger.info("deleted audit log %s", item_id)


class SettingService:
    """Business rules for settings."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_setting(self, data: dict) -> dict:
        """Validate and store a new setting."""
        item = Setting.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid setting: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created setting %s", item.id)
        return item.to_dict()

    def get_setting(self, item_id: int) -> dict:
        """Return one setting or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"setting {item_id} not found")
        return item.to_dict()

    def list_settings(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of settings with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_setting(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing setting."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"setting {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated setting %s", item_id)
        return item.to_dict()

    def delete_setting(self, item_id: int) -> None:
        """Delete a setting or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"setting {item_id} not found")
        logger.info("deleted setting %s", item_id)


class RecipeService:
    """Business rules for recipes."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_recipe(self, data: dict) -> dict:
        """Validate and store a new recipe."""
        item = Recipe.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid recipe: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created recipe %s", item.id)
        return item.to_dict()

    def get_recipe(self, item_id: int) -> dict:
        """Return one recipe or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"recipe {item_id} not found")
        return item.to_dict()

    def list_recipes(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of recipes with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_recipe(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing recipe."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"recipe {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated recipe %s", item_id)
        return item.to_dict()

    def delete_recipe(self, item_id: int) -> None:
        """Delete a recipe or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"recipe {item_id} not found")
        logger.info("deleted recipe %s", item_id)


class IngredientService:
    """Business rules for ingredients."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_ingredient(self, data: dict) -> dict:
        """Validate and store a new ingredient."""
        item = Ingredient.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid ingredient: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created ingredient %s", item.id)
        return item.to_dict()

    def get_ingredient(self, item_id: int) -> dict:
        """Return one ingredient or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"ingredient {item_id} not found")
        return item.to_dict()

    def list_ingredients(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of ingredients with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_ingredient(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing ingredient."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"ingredient {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated ingredient %s", item_id)
        return item.to_dict()

    def delete_ingredient(self, item_id: int) -> None:
        """Delete a ingredient or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"ingredient {item_id} not found")
        logger.info("deleted ingredient %s", item_id)


class PlaylistService:
    """Business rules for playlists."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_playlist(self, data: dict) -> dict:
        """Validate and store a new playlist."""
        item = Playlist.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid playlist: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created playlist %s", item.id)
        return item.to_dict()

    def get_playlist(self, item_id: int) -> dict:
        """Return one playlist or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"playlist {item_id} not found")
        return item.to_dict()

    def list_playlists(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of playlists with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_playlist(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing playlist."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"playlist {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated playlist %s", item_id)
        return item.to_dict()

    def delete_playlist(self, item_id: int) -> None:
        """Delete a playlist or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"playlist {item_id} not found")
        logger.info("deleted playlist %s", item_id)


class TrackService:
    """Business rules for tracks."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_track(self, data: dict) -> dict:
        """Validate and store a new track."""
        item = Track.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid track: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created track %s", item.id)
        return item.to_dict()

    def get_track(self, item_id: int) -> dict:
        """Return one track or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"track {item_id} not found")
        return item.to_dict()

    def list_tracks(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of tracks with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_track(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing track."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"track {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated track %s", item_id)
        return item.to_dict()

    def delete_track(self, item_id: int) -> None:
        """Delete a track or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"track {item_id} not found")
        logger.info("deleted track %s", item_id)


class DeviceService:
    """Business rules for devices."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def create_device(self, data: dict) -> dict:
        """Validate and store a new device."""
        item = Device.from_dict(data)
        errors = item.validate()
        if errors:
            logger.warning("invalid device: %s", errors)
            raise ValidationError(errors)
        item = self.repository.add(item)
        logger.info("created device %s", item.id)
        return item.to_dict()

    def get_device(self, item_id: int) -> dict:
        """Return one device or raise NotFoundError."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"device {item_id} not found")
        return item.to_dict()

    def list_devices(self, page: int = 1, page_size: int = 20) -> dict:
        """Return one page of devices with the total count."""
        if page < 1:
            page = 1
        offset = (page - 1) * page_size
        items = self.repository.list(limit=page_size, offset=offset)
        return {
            "items": [item.to_dict() for item in items],
            "page": page,
            "page_size": page_size,
            "total": self.repository.count(),
        }

    def update_device(self, item_id: int, changes: dict) -> dict:
        """Apply changes to an existing device."""
        item = self.repository.get(item_id)
        if item is None:
            raise NotFoundError(f"device {item_id} not found")
        for key, value in changes.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        errors = item.validate()
        if errors:
            raise ValidationError(errors)
        self.repository.update(item)
        logger.info("updated device %s", item_id)
        return item.to_dict()

    def delete_device(self, item_id: int) -> None:
        """Delete a device or raise NotFoundError."""
        if not self.repository.delete(item_id):
            raise NotFoundError(f"device {item_id} not found")
        logger.info("deleted device %s", item_id)

