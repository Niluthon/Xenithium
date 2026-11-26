"""Domain models for the store, the office and the clinic back ends."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


@dataclass
class User:
    """A user record."""

    id: Optional[int] = None
    username: str = ""
    email: str = ""
    full_name: str = ""
    is_active: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this user; empty when it is valid."""
        errors: list[str] = []
        if not self.username:
            errors.append("username is required")
        if not self.email:
            errors.append("email is required")
        elif "@" not in self.email:
            errors.append("email must be an email address")
        if not self.full_name:
            errors.append("full_name is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            username=str(data.get("username", "")),
            email=str(data.get("email", "")),
            full_name=str(data.get("full_name", "")),
            is_active=bool(data.get("is_active", False)),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class Product:
    """A product record."""

    id: Optional[int] = None
    sku: str = ""
    name: str = ""
    description: str = ""
    price: float = 0.0
    stock: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this product; empty when it is valid."""
        errors: list[str] = []
        if not self.sku:
            errors.append("sku is required")
        if not self.name:
            errors.append("name is required")
        if not self.description:
            errors.append("description is required")
        if self.price < 0:
            errors.append("price must not be negative")
        if self.stock < 0:
            errors.append("stock must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Product":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            sku=str(data.get("sku", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            price=float(data.get("price", 0.0)),
            stock=int(data.get("stock", 0)),
        )


@dataclass
class Order:
    """A order record."""

    id: Optional[int] = None
    customer_id: int = 0
    status: str = ""
    total: float = 0.0
    currency: str = ""
    placed_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this order; empty when it is valid."""
        errors: list[str] = []
        if self.customer_id < 0:
            errors.append("customer_id must not be negative")
        if not self.status:
            errors.append("status is required")
        if self.total < 0:
            errors.append("total must not be negative")
        if not self.currency:
            errors.append("currency is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["placed_at"] = self.placed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Order":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            customer_id=int(data.get("customer_id", 0)),
            status=str(data.get("status", "")),
            total=float(data.get("total", 0.0)),
            currency=str(data.get("currency", "")),
            placed_at=datetime.fromisoformat(data["placed_at"]) if "placed_at" in data else datetime.now(),
        )


@dataclass
class OrderItem:
    """A order item record."""

    id: Optional[int] = None
    order_id: int = 0
    product_id: int = 0
    quantity: int = 0
    unit_price: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of problems with this order item; empty when it is valid."""
        errors: list[str] = []
        if self.order_id < 0:
            errors.append("order_id must not be negative")
        if self.product_id < 0:
            errors.append("product_id must not be negative")
        if self.quantity < 0:
            errors.append("quantity must not be negative")
        if self.unit_price < 0:
            errors.append("unit_price must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderItem":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            order_id=int(data.get("order_id", 0)),
            product_id=int(data.get("product_id", 0)),
            quantity=int(data.get("quantity", 0)),
            unit_price=float(data.get("unit_price", 0.0)),
        )


@dataclass
class Customer:
    """A customer record."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    country: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this customer; empty when it is valid."""
        errors: list[str] = []
        if not self.first_name:
            errors.append("first_name is required")
        if not self.last_name:
            errors.append("last_name is required")
        if not self.email:
            errors.append("email is required")
        elif "@" not in self.email:
            errors.append("email must be an email address")
        if not self.phone:
            errors.append("phone is required")
        if not self.country:
            errors.append("country is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Customer":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            email=str(data.get("email", "")),
            phone=str(data.get("phone", "")),
            country=str(data.get("country", "")),
        )


@dataclass
class Invoice:
    """A invoice record."""

    id: Optional[int] = None
    order_id: int = 0
    number: str = ""
    amount: float = 0.0
    due_date: datetime = field(default_factory=datetime.now)
    paid: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this invoice; empty when it is valid."""
        errors: list[str] = []
        if self.order_id < 0:
            errors.append("order_id must not be negative")
        if not self.number:
            errors.append("number is required")
        if self.amount < 0:
            errors.append("amount must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["due_date"] = self.due_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Invoice":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            order_id=int(data.get("order_id", 0)),
            number=str(data.get("number", "")),
            amount=float(data.get("amount", 0.0)),
            due_date=datetime.fromisoformat(data["due_date"]) if "due_date" in data else datetime.now(),
            paid=bool(data.get("paid", False)),
        )


@dataclass
class Payment:
    """A payment record."""

    id: Optional[int] = None
    invoice_id: int = 0
    method: str = ""
    amount: float = 0.0
    reference: str = ""
    received_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this payment; empty when it is valid."""
        errors: list[str] = []
        if self.invoice_id < 0:
            errors.append("invoice_id must not be negative")
        if not self.method:
            errors.append("method is required")
        if self.amount < 0:
            errors.append("amount must not be negative")
        if not self.reference:
            errors.append("reference is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["received_at"] = self.received_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Payment":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            invoice_id=int(data.get("invoice_id", 0)),
            method=str(data.get("method", "")),
            amount=float(data.get("amount", 0.0)),
            reference=str(data.get("reference", "")),
            received_at=datetime.fromisoformat(data["received_at"]) if "received_at" in data else datetime.now(),
        )


@dataclass
class Category:
    """A category record."""

    id: Optional[int] = None
    name: str = ""
    slug: str = ""
    parent_id: int = 0
    position: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this category; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.slug:
            errors.append("slug is required")
        if self.parent_id < 0:
            errors.append("parent_id must not be negative")
        if self.position < 0:
            errors.append("position must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Category":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            slug=str(data.get("slug", "")),
            parent_id=int(data.get("parent_id", 0)),
            position=int(data.get("position", 0)),
        )


@dataclass
class Review:
    """A review record."""

    id: Optional[int] = None
    product_id: int = 0
    user_id: int = 0
    rating: int = 0
    title: str = ""
    body: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this review; empty when it is valid."""
        errors: list[str] = []
        if self.product_id < 0:
            errors.append("product_id must not be negative")
        if self.user_id < 0:
            errors.append("user_id must not be negative")
        if self.rating < 0:
            errors.append("rating must not be negative")
        if not self.title:
            errors.append("title is required")
        if not self.body:
            errors.append("body is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Review":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            product_id=int(data.get("product_id", 0)),
            user_id=int(data.get("user_id", 0)),
            rating=int(data.get("rating", 0)),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
        )


@dataclass
class Address:
    """A address record."""

    id: Optional[int] = None
    customer_id: int = 0
    street: str = ""
    city: str = ""
    postal_code: str = ""
    country: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this address; empty when it is valid."""
        errors: list[str] = []
        if self.customer_id < 0:
            errors.append("customer_id must not be negative")
        if not self.street:
            errors.append("street is required")
        if not self.city:
            errors.append("city is required")
        if not self.postal_code:
            errors.append("postal_code is required")
        if not self.country:
            errors.append("country is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Address":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            customer_id=int(data.get("customer_id", 0)),
            street=str(data.get("street", "")),
            city=str(data.get("city", "")),
            postal_code=str(data.get("postal_code", "")),
            country=str(data.get("country", "")),
        )


@dataclass
class Shipment:
    """A shipment record."""

    id: Optional[int] = None
    order_id: int = 0
    carrier: str = ""
    tracking_number: str = ""
    shipped_at: datetime = field(default_factory=datetime.now)
    delivered: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this shipment; empty when it is valid."""
        errors: list[str] = []
        if self.order_id < 0:
            errors.append("order_id must not be negative")
        if not self.carrier:
            errors.append("carrier is required")
        if not self.tracking_number:
            errors.append("tracking_number is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["shipped_at"] = self.shipped_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Shipment":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            order_id=int(data.get("order_id", 0)),
            carrier=str(data.get("carrier", "")),
            tracking_number=str(data.get("tracking_number", "")),
            shipped_at=datetime.fromisoformat(data["shipped_at"]) if "shipped_at" in data else datetime.now(),
            delivered=bool(data.get("delivered", False)),
        )


@dataclass
class Warehouse:
    """A warehouse record."""

    id: Optional[int] = None
    name: str = ""
    city: str = ""
    capacity: int = 0
    is_open: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this warehouse; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.city:
            errors.append("city is required")
        if self.capacity < 0:
            errors.append("capacity must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Warehouse":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            city=str(data.get("city", "")),
            capacity=int(data.get("capacity", 0)),
            is_open=bool(data.get("is_open", False)),
        )


@dataclass
class Supplier:
    """A supplier record."""

    id: Optional[int] = None
    name: str = ""
    contact_email: str = ""
    phone: str = ""
    country: str = ""
    rating: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this supplier; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.contact_email:
            errors.append("contact_email is required")
        elif "@" not in self.contact_email:
            errors.append("contact_email must be an email address")
        if not self.phone:
            errors.append("phone is required")
        if not self.country:
            errors.append("country is required")
        if self.rating < 0:
            errors.append("rating must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Supplier":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            contact_email=str(data.get("contact_email", "")),
            phone=str(data.get("phone", "")),
            country=str(data.get("country", "")),
            rating=int(data.get("rating", 0)),
        )


@dataclass
class Coupon:
    """A coupon record."""

    id: Optional[int] = None
    code: str = ""
    discount_percent: int = 0
    valid_until: datetime = field(default_factory=datetime.now)
    max_uses: int = 0
    is_active: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this coupon; empty when it is valid."""
        errors: list[str] = []
        if not self.code:
            errors.append("code is required")
        if self.discount_percent < 0:
            errors.append("discount_percent must not be negative")
        if self.max_uses < 0:
            errors.append("max_uses must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["valid_until"] = self.valid_until.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Coupon":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            code=str(data.get("code", "")),
            discount_percent=int(data.get("discount_percent", 0)),
            valid_until=datetime.fromisoformat(data["valid_until"]) if "valid_until" in data else datetime.now(),
            max_uses=int(data.get("max_uses", 0)),
            is_active=bool(data.get("is_active", False)),
        )


@dataclass
class Employee:
    """A employee record."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    department: str = ""
    salary: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of problems with this employee; empty when it is valid."""
        errors: list[str] = []
        if not self.first_name:
            errors.append("first_name is required")
        if not self.last_name:
            errors.append("last_name is required")
        if not self.email:
            errors.append("email is required")
        elif "@" not in self.email:
            errors.append("email must be an email address")
        if not self.department:
            errors.append("department is required")
        if self.salary < 0:
            errors.append("salary must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Employee":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            email=str(data.get("email", "")),
            department=str(data.get("department", "")),
            salary=float(data.get("salary", 0.0)),
        )


@dataclass
class Department:
    """A department record."""

    id: Optional[int] = None
    name: str = ""
    code: str = ""
    manager_id: int = 0
    budget: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of problems with this department; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.code:
            errors.append("code is required")
        if self.manager_id < 0:
            errors.append("manager_id must not be negative")
        if self.budget < 0:
            errors.append("budget must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Department":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            code=str(data.get("code", "")),
            manager_id=int(data.get("manager_id", 0)),
            budget=float(data.get("budget", 0.0)),
        )


@dataclass
class Project:
    """A project record."""

    id: Optional[int] = None
    name: str = ""
    description: str = ""
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=datetime.now)
    budget: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of problems with this project; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.description:
            errors.append("description is required")
        if self.budget < 0:
            errors.append("budget must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            start_date=datetime.fromisoformat(data["start_date"]) if "start_date" in data else datetime.now(),
            end_date=datetime.fromisoformat(data["end_date"]) if "end_date" in data else datetime.now(),
            budget=float(data.get("budget", 0.0)),
        )


@dataclass
class Task:
    """A task record."""

    id: Optional[int] = None
    project_id: int = 0
    title: str = ""
    status: str = ""
    priority: int = 0
    due_date: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this task; empty when it is valid."""
        errors: list[str] = []
        if self.project_id < 0:
            errors.append("project_id must not be negative")
        if not self.title:
            errors.append("title is required")
        if not self.status:
            errors.append("status is required")
        if self.priority < 0:
            errors.append("priority must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["due_date"] = self.due_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            project_id=int(data.get("project_id", 0)),
            title=str(data.get("title", "")),
            status=str(data.get("status", "")),
            priority=int(data.get("priority", 0)),
            due_date=datetime.fromisoformat(data["due_date"]) if "due_date" in data else datetime.now(),
        )


@dataclass
class Comment:
    """A comment record."""

    id: Optional[int] = None
    task_id: int = 0
    author_id: int = 0
    body: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this comment; empty when it is valid."""
        errors: list[str] = []
        if self.task_id < 0:
            errors.append("task_id must not be negative")
        if self.author_id < 0:
            errors.append("author_id must not be negative")
        if not self.body:
            errors.append("body is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Comment":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            task_id=int(data.get("task_id", 0)),
            author_id=int(data.get("author_id", 0)),
            body=str(data.get("body", "")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class Tag:
    """A tag record."""

    id: Optional[int] = None
    name: str = ""
    color: str = ""
    usage_count: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this tag; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.color:
            errors.append("color is required")
        if self.usage_count < 0:
            errors.append("usage_count must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tag":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            color=str(data.get("color", "")),
            usage_count=int(data.get("usage_count", 0)),
        )


@dataclass
class Article:
    """A article record."""

    id: Optional[int] = None
    title: str = ""
    slug: str = ""
    body: str = ""
    author_id: int = 0
    published: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this article; empty when it is valid."""
        errors: list[str] = []
        if not self.title:
            errors.append("title is required")
        if not self.slug:
            errors.append("slug is required")
        if not self.body:
            errors.append("body is required")
        if self.author_id < 0:
            errors.append("author_id must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Article":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            title=str(data.get("title", "")),
            slug=str(data.get("slug", "")),
            body=str(data.get("body", "")),
            author_id=int(data.get("author_id", 0)),
            published=bool(data.get("published", False)),
        )


@dataclass
class Author:
    """A author record."""

    id: Optional[int] = None
    name: str = ""
    bio: str = ""
    email: str = ""
    website: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this author; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.bio:
            errors.append("bio is required")
        if not self.email:
            errors.append("email is required")
        elif "@" not in self.email:
            errors.append("email must be an email address")
        if not self.website:
            errors.append("website is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Author":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            bio=str(data.get("bio", "")),
            email=str(data.get("email", "")),
            website=str(data.get("website", "")),
        )


@dataclass
class Book:
    """A book record."""

    id: Optional[int] = None
    title: str = ""
    isbn: str = ""
    author_id: int = 0
    year: int = 0
    pages: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this book; empty when it is valid."""
        errors: list[str] = []
        if not self.title:
            errors.append("title is required")
        if not self.isbn:
            errors.append("isbn is required")
        if self.author_id < 0:
            errors.append("author_id must not be negative")
        if self.year < 0:
            errors.append("year must not be negative")
        if self.pages < 0:
            errors.append("pages must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Book":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            title=str(data.get("title", "")),
            isbn=str(data.get("isbn", "")),
            author_id=int(data.get("author_id", 0)),
            year=int(data.get("year", 0)),
            pages=int(data.get("pages", 0)),
        )


@dataclass
class Library:
    """A library record."""

    id: Optional[int] = None
    name: str = ""
    city: str = ""
    opening_hour: int = 0
    closing_hour: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this library; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.city:
            errors.append("city is required")
        if self.opening_hour < 0:
            errors.append("opening_hour must not be negative")
        if self.closing_hour < 0:
            errors.append("closing_hour must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Library":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            city=str(data.get("city", "")),
            opening_hour=int(data.get("opening_hour", 0)),
            closing_hour=int(data.get("closing_hour", 0)),
        )


@dataclass
class Loan:
    """A loan record."""

    id: Optional[int] = None
    book_id: int = 0
    member_id: int = 0
    borrowed_at: datetime = field(default_factory=datetime.now)
    due_date: datetime = field(default_factory=datetime.now)
    returned: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this loan; empty when it is valid."""
        errors: list[str] = []
        if self.book_id < 0:
            errors.append("book_id must not be negative")
        if self.member_id < 0:
            errors.append("member_id must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["borrowed_at"] = self.borrowed_at.isoformat()
        data["due_date"] = self.due_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Loan":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            book_id=int(data.get("book_id", 0)),
            member_id=int(data.get("member_id", 0)),
            borrowed_at=datetime.fromisoformat(data["borrowed_at"]) if "borrowed_at" in data else datetime.now(),
            due_date=datetime.fromisoformat(data["due_date"]) if "due_date" in data else datetime.now(),
            returned=bool(data.get("returned", False)),
        )


@dataclass
class Member:
    """A member record."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    joined_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this member; empty when it is valid."""
        errors: list[str] = []
        if not self.first_name:
            errors.append("first_name is required")
        if not self.last_name:
            errors.append("last_name is required")
        if not self.email:
            errors.append("email is required")
        elif "@" not in self.email:
            errors.append("email must be an email address")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["joined_at"] = self.joined_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Member":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            email=str(data.get("email", "")),
            joined_at=datetime.fromisoformat(data["joined_at"]) if "joined_at" in data else datetime.now(),
        )


@dataclass
class Event:
    """A event record."""

    id: Optional[int] = None
    title: str = ""
    location: str = ""
    starts_at: datetime = field(default_factory=datetime.now)
    ends_at: datetime = field(default_factory=datetime.now)
    capacity: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this event; empty when it is valid."""
        errors: list[str] = []
        if not self.title:
            errors.append("title is required")
        if not self.location:
            errors.append("location is required")
        if self.capacity < 0:
            errors.append("capacity must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["starts_at"] = self.starts_at.isoformat()
        data["ends_at"] = self.ends_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            title=str(data.get("title", "")),
            location=str(data.get("location", "")),
            starts_at=datetime.fromisoformat(data["starts_at"]) if "starts_at" in data else datetime.now(),
            ends_at=datetime.fromisoformat(data["ends_at"]) if "ends_at" in data else datetime.now(),
            capacity=int(data.get("capacity", 0)),
        )


@dataclass
class Ticket:
    """A ticket record."""

    id: Optional[int] = None
    event_id: int = 0
    holder_name: str = ""
    seat: str = ""
    price: float = 0.0
    checked_in: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this ticket; empty when it is valid."""
        errors: list[str] = []
        if self.event_id < 0:
            errors.append("event_id must not be negative")
        if not self.holder_name:
            errors.append("holder_name is required")
        if not self.seat:
            errors.append("seat is required")
        if self.price < 0:
            errors.append("price must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ticket":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            event_id=int(data.get("event_id", 0)),
            holder_name=str(data.get("holder_name", "")),
            seat=str(data.get("seat", "")),
            price=float(data.get("price", 0.0)),
            checked_in=bool(data.get("checked_in", False)),
        )


@dataclass
class Venue:
    """A venue record."""

    id: Optional[int] = None
    name: str = ""
    address: str = ""
    city: str = ""
    capacity: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this venue; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.address:
            errors.append("address is required")
        if not self.city:
            errors.append("city is required")
        if self.capacity < 0:
            errors.append("capacity must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Venue":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            address=str(data.get("address", "")),
            city=str(data.get("city", "")),
            capacity=int(data.get("capacity", 0)),
        )


@dataclass
class Vehicle:
    """A vehicle record."""

    id: Optional[int] = None
    plate: str = ""
    make: str = ""
    model: str = ""
    year: int = 0
    mileage: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this vehicle; empty when it is valid."""
        errors: list[str] = []
        if not self.plate:
            errors.append("plate is required")
        if not self.make:
            errors.append("make is required")
        if not self.model:
            errors.append("model is required")
        if self.year < 0:
            errors.append("year must not be negative")
        if self.mileage < 0:
            errors.append("mileage must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vehicle":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            plate=str(data.get("plate", "")),
            make=str(data.get("make", "")),
            model=str(data.get("model", "")),
            year=int(data.get("year", 0)),
            mileage=int(data.get("mileage", 0)),
        )


@dataclass
class Driver:
    """A driver record."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    license_number: str = ""
    rating: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of problems with this driver; empty when it is valid."""
        errors: list[str] = []
        if not self.first_name:
            errors.append("first_name is required")
        if not self.last_name:
            errors.append("last_name is required")
        if not self.license_number:
            errors.append("license_number is required")
        if self.rating < 0:
            errors.append("rating must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Driver":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            license_number=str(data.get("license_number", "")),
            rating=float(data.get("rating", 0.0)),
        )


@dataclass
class Trip:
    """A trip record."""

    id: Optional[int] = None
    vehicle_id: int = 0
    driver_id: int = 0
    distance_km: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    fare: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of problems with this trip; empty when it is valid."""
        errors: list[str] = []
        if self.vehicle_id < 0:
            errors.append("vehicle_id must not be negative")
        if self.driver_id < 0:
            errors.append("driver_id must not be negative")
        if self.distance_km < 0:
            errors.append("distance_km must not be negative")
        if self.fare < 0:
            errors.append("fare must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trip":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            vehicle_id=int(data.get("vehicle_id", 0)),
            driver_id=int(data.get("driver_id", 0)),
            distance_km=float(data.get("distance_km", 0.0)),
            started_at=datetime.fromisoformat(data["started_at"]) if "started_at" in data else datetime.now(),
            fare=float(data.get("fare", 0.0)),
        )


@dataclass
class Patient:
    """A patient record."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    birth_date: datetime = field(default_factory=datetime.now)
    phone: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this patient; empty when it is valid."""
        errors: list[str] = []
        if not self.first_name:
            errors.append("first_name is required")
        if not self.last_name:
            errors.append("last_name is required")
        if not self.phone:
            errors.append("phone is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["birth_date"] = self.birth_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Patient":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            birth_date=datetime.fromisoformat(data["birth_date"]) if "birth_date" in data else datetime.now(),
            phone=str(data.get("phone", "")),
        )


@dataclass
class Doctor:
    """A doctor record."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    specialty: str = ""
    room: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this doctor; empty when it is valid."""
        errors: list[str] = []
        if not self.first_name:
            errors.append("first_name is required")
        if not self.last_name:
            errors.append("last_name is required")
        if not self.specialty:
            errors.append("specialty is required")
        if not self.room:
            errors.append("room is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Doctor":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            specialty=str(data.get("specialty", "")),
            room=str(data.get("room", "")),
        )


@dataclass
class Appointment:
    """A appointment record."""

    id: Optional[int] = None
    patient_id: int = 0
    doctor_id: int = 0
    scheduled_at: datetime = field(default_factory=datetime.now)
    reason: str = ""
    confirmed: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this appointment; empty when it is valid."""
        errors: list[str] = []
        if self.patient_id < 0:
            errors.append("patient_id must not be negative")
        if self.doctor_id < 0:
            errors.append("doctor_id must not be negative")
        if not self.reason:
            errors.append("reason is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["scheduled_at"] = self.scheduled_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Appointment":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            patient_id=int(data.get("patient_id", 0)),
            doctor_id=int(data.get("doctor_id", 0)),
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if "scheduled_at" in data else datetime.now(),
            reason=str(data.get("reason", "")),
            confirmed=bool(data.get("confirmed", False)),
        )


@dataclass
class Course:
    """A course record."""

    id: Optional[int] = None
    title: str = ""
    code: str = ""
    credits: int = 0
    teacher_id: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this course; empty when it is valid."""
        errors: list[str] = []
        if not self.title:
            errors.append("title is required")
        if not self.code:
            errors.append("code is required")
        if self.credits < 0:
            errors.append("credits must not be negative")
        if self.teacher_id < 0:
            errors.append("teacher_id must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Course":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            title=str(data.get("title", "")),
            code=str(data.get("code", "")),
            credits=int(data.get("credits", 0)),
            teacher_id=int(data.get("teacher_id", 0)),
        )


@dataclass
class Student:
    """A student record."""

    id: Optional[int] = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    enrolled_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this student; empty when it is valid."""
        errors: list[str] = []
        if not self.first_name:
            errors.append("first_name is required")
        if not self.last_name:
            errors.append("last_name is required")
        if not self.email:
            errors.append("email is required")
        elif "@" not in self.email:
            errors.append("email must be an email address")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["enrolled_at"] = self.enrolled_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Student":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            email=str(data.get("email", "")),
            enrolled_at=datetime.fromisoformat(data["enrolled_at"]) if "enrolled_at" in data else datetime.now(),
        )


@dataclass
class Enrollment:
    """A enrollment record."""

    id: Optional[int] = None
    student_id: int = 0
    course_id: int = 0
    grade: float = 0.0
    completed: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this enrollment; empty when it is valid."""
        errors: list[str] = []
        if self.student_id < 0:
            errors.append("student_id must not be negative")
        if self.course_id < 0:
            errors.append("course_id must not be negative")
        if self.grade < 0:
            errors.append("grade must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Enrollment":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            student_id=int(data.get("student_id", 0)),
            course_id=int(data.get("course_id", 0)),
            grade=float(data.get("grade", 0.0)),
            completed=bool(data.get("completed", False)),
        )


@dataclass
class Room:
    """A room record."""

    id: Optional[int] = None
    number: str = ""
    floor: int = 0
    beds: int = 0
    price_per_night: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of problems with this room; empty when it is valid."""
        errors: list[str] = []
        if not self.number:
            errors.append("number is required")
        if self.floor < 0:
            errors.append("floor must not be negative")
        if self.beds < 0:
            errors.append("beds must not be negative")
        if self.price_per_night < 0:
            errors.append("price_per_night must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Room":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            number=str(data.get("number", "")),
            floor=int(data.get("floor", 0)),
            beds=int(data.get("beds", 0)),
            price_per_night=float(data.get("price_per_night", 0.0)),
        )


@dataclass
class Booking:
    """A booking record."""

    id: Optional[int] = None
    room_id: int = 0
    guest_name: str = ""
    check_in: datetime = field(default_factory=datetime.now)
    check_out: datetime = field(default_factory=datetime.now)
    paid: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this booking; empty when it is valid."""
        errors: list[str] = []
        if self.room_id < 0:
            errors.append("room_id must not be negative")
        if not self.guest_name:
            errors.append("guest_name is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["check_in"] = self.check_in.isoformat()
        data["check_out"] = self.check_out.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Booking":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            room_id=int(data.get("room_id", 0)),
            guest_name=str(data.get("guest_name", "")),
            check_in=datetime.fromisoformat(data["check_in"]) if "check_in" in data else datetime.now(),
            check_out=datetime.fromisoformat(data["check_out"]) if "check_out" in data else datetime.now(),
            paid=bool(data.get("paid", False)),
        )


@dataclass
class Subscription:
    """A subscription record."""

    id: Optional[int] = None
    user_id: int = 0
    plan: str = ""
    price: float = 0.0
    renews_at: datetime = field(default_factory=datetime.now)
    is_active: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this subscription; empty when it is valid."""
        errors: list[str] = []
        if self.user_id < 0:
            errors.append("user_id must not be negative")
        if not self.plan:
            errors.append("plan is required")
        if self.price < 0:
            errors.append("price must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["renews_at"] = self.renews_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subscription":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            user_id=int(data.get("user_id", 0)),
            plan=str(data.get("plan", "")),
            price=float(data.get("price", 0.0)),
            renews_at=datetime.fromisoformat(data["renews_at"]) if "renews_at" in data else datetime.now(),
            is_active=bool(data.get("is_active", False)),
        )


@dataclass
class Notification:
    """A notification record."""

    id: Optional[int] = None
    user_id: int = 0
    title: str = ""
    message: str = ""
    read: bool = False
    sent_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this notification; empty when it is valid."""
        errors: list[str] = []
        if self.user_id < 0:
            errors.append("user_id must not be negative")
        if not self.title:
            errors.append("title is required")
        if not self.message:
            errors.append("message is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["sent_at"] = self.sent_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Notification":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            user_id=int(data.get("user_id", 0)),
            title=str(data.get("title", "")),
            message=str(data.get("message", "")),
            read=bool(data.get("read", False)),
            sent_at=datetime.fromisoformat(data["sent_at"]) if "sent_at" in data else datetime.now(),
        )


@dataclass
class Session:
    """A session record."""

    id: Optional[int] = None
    user_id: int = 0
    token: str = ""
    expires_at: datetime = field(default_factory=datetime.now)
    ip_address: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this session; empty when it is valid."""
        errors: list[str] = []
        if self.user_id < 0:
            errors.append("user_id must not be negative")
        if not self.token:
            errors.append("token is required")
        if not self.ip_address:
            errors.append("ip_address is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["expires_at"] = self.expires_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            user_id=int(data.get("user_id", 0)),
            token=str(data.get("token", "")),
            expires_at=datetime.fromisoformat(data["expires_at"]) if "expires_at" in data else datetime.now(),
            ip_address=str(data.get("ip_address", "")),
        )


@dataclass
class AuditLog:
    """A audit log record."""

    id: Optional[int] = None
    actor_id: int = 0
    action: str = ""
    target: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this audit log; empty when it is valid."""
        errors: list[str] = []
        if self.actor_id < 0:
            errors.append("actor_id must not be negative")
        if not self.action:
            errors.append("action is required")
        if not self.target:
            errors.append("target is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditLog":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            actor_id=int(data.get("actor_id", 0)),
            action=str(data.get("action", "")),
            target=str(data.get("target", "")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class Setting:
    """A setting record."""

    id: Optional[int] = None
    key: str = ""
    value: str = ""
    description: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this setting; empty when it is valid."""
        errors: list[str] = []
        if not self.key:
            errors.append("key is required")
        if not self.value:
            errors.append("value is required")
        if not self.description:
            errors.append("description is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Setting":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            key=str(data.get("key", "")),
            value=str(data.get("value", "")),
            description=str(data.get("description", "")),
        )


@dataclass
class Recipe:
    """A recipe record."""

    id: Optional[int] = None
    title: str = ""
    servings: int = 0
    prep_minutes: int = 0
    cook_minutes: int = 0
    vegetarian: bool = False

    def validate(self) -> list[str]:
        """Return a list of problems with this recipe; empty when it is valid."""
        errors: list[str] = []
        if not self.title:
            errors.append("title is required")
        if self.servings < 0:
            errors.append("servings must not be negative")
        if self.prep_minutes < 0:
            errors.append("prep_minutes must not be negative")
        if self.cook_minutes < 0:
            errors.append("cook_minutes must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            title=str(data.get("title", "")),
            servings=int(data.get("servings", 0)),
            prep_minutes=int(data.get("prep_minutes", 0)),
            cook_minutes=int(data.get("cook_minutes", 0)),
            vegetarian=bool(data.get("vegetarian", False)),
        )


@dataclass
class Ingredient:
    """A ingredient record."""

    id: Optional[int] = None
    recipe_id: int = 0
    name: str = ""
    amount: float = 0.0
    unit: str = ""

    def validate(self) -> list[str]:
        """Return a list of problems with this ingredient; empty when it is valid."""
        errors: list[str] = []
        if self.recipe_id < 0:
            errors.append("recipe_id must not be negative")
        if not self.name:
            errors.append("name is required")
        if self.amount < 0:
            errors.append("amount must not be negative")
        if not self.unit:
            errors.append("unit is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ingredient":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            recipe_id=int(data.get("recipe_id", 0)),
            name=str(data.get("name", "")),
            amount=float(data.get("amount", 0.0)),
            unit=str(data.get("unit", "")),
        )


@dataclass
class Playlist:
    """A playlist record."""

    id: Optional[int] = None
    name: str = ""
    owner_id: int = 0
    is_public: bool = False
    track_count: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this playlist; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if self.owner_id < 0:
            errors.append("owner_id must not be negative")
        if self.track_count < 0:
            errors.append("track_count must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Playlist":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            owner_id=int(data.get("owner_id", 0)),
            is_public=bool(data.get("is_public", False)),
            track_count=int(data.get("track_count", 0)),
        )


@dataclass
class Track:
    """A track record."""

    id: Optional[int] = None
    title: str = ""
    artist: str = ""
    album: str = ""
    duration_seconds: int = 0

    def validate(self) -> list[str]:
        """Return a list of problems with this track; empty when it is valid."""
        errors: list[str] = []
        if not self.title:
            errors.append("title is required")
        if not self.artist:
            errors.append("artist is required")
        if not self.album:
            errors.append("album is required")
        if self.duration_seconds < 0:
            errors.append("duration_seconds must not be negative")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            title=str(data.get("title", "")),
            artist=str(data.get("artist", "")),
            album=str(data.get("album", "")),
            duration_seconds=int(data.get("duration_seconds", 0)),
        )


@dataclass
class Device:
    """A device record."""

    id: Optional[int] = None
    name: str = ""
    serial_number: str = ""
    firmware: str = ""
    online: bool = False
    last_seen: datetime = field(default_factory=datetime.now)

    def validate(self) -> list[str]:
        """Return a list of problems with this device; empty when it is valid."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.serial_number:
            errors.append("serial_number is required")
        if not self.firmware:
            errors.append("firmware is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        data = asdict(self)
        data["last_seen"] = self.last_seen.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Device":
        """Build a record from a dictionary, as produced by to_dict."""
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            serial_number=str(data.get("serial_number", "")),
            firmware=str(data.get("firmware", "")),
            online=bool(data.get("online", False)),
            last_seen=datetime.fromisoformat(data["last_seen"]) if "last_seen" in data else datetime.now(),
        )

