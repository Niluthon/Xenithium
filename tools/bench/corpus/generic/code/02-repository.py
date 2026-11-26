"""SQLite repositories: one class per table with the usual create, read, update and delete methods."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from models import User, Product, Order, OrderItem, Customer, Invoice, Payment, Category  # noqa: F401


class BaseRepository:
    """Shared connection handling for every repository."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    def commit(self) -> None:
        self.connection.commit()


class UserRepository(BaseRepository):
    """Reads and writes users in the users table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                full_name TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: User) -> User:
        """Insert a new user and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO users (username, email, full_name, is_active, created_at) VALUES (?, ?, ?, ?, ?)",
            (item.username, item.email, item.full_name, item.is_active, item.created_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[User]:
        """Return the user with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM users WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return User.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[User]:
        """Return up to limit users, ordered by id."""
        rows = self.connection.execute("SELECT * FROM users ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [User.from_dict(dict(row)) for row in rows]

    def update(self, item: User) -> None:
        """Write every field of an existing user."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE users SET username = ?, email = ?, full_name = ?, is_active = ?, created_at = ? WHERE id = ?",
            (item.username, item.email, item.full_name, item.is_active, item.created_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the user with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM users WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of users."""
        return self.connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]


class ProductRepository(BaseRepository):
    """Reads and writes products in the products table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Product) -> Product:
        """Insert a new product and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO products (sku, name, description, price, stock) VALUES (?, ?, ?, ?, ?)",
            (item.sku, item.name, item.description, item.price, item.stock,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Product]:
        """Return the product with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM products WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Product.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Product]:
        """Return up to limit products, ordered by id."""
        rows = self.connection.execute("SELECT * FROM products ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Product.from_dict(dict(row)) for row in rows]

    def update(self, item: Product) -> None:
        """Write every field of an existing product."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE products SET sku = ?, name = ?, description = ?, price = ?, stock = ? WHERE id = ?",
            (item.sku, item.name, item.description, item.price, item.stock, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the product with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM products WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of products."""
        return self.connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]


class OrderRepository(BaseRepository):
    """Reads and writes orders in the orders table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                total REAL NOT NULL,
                currency TEXT NOT NULL,
                placed_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Order) -> Order:
        """Insert a new order and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO orders (customer_id, status, total, currency, placed_at) VALUES (?, ?, ?, ?, ?)",
            (item.customer_id, item.status, item.total, item.currency, item.placed_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Order]:
        """Return the order with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM orders WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Order.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Order]:
        """Return up to limit orders, ordered by id."""
        rows = self.connection.execute("SELECT * FROM orders ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Order.from_dict(dict(row)) for row in rows]

    def update(self, item: Order) -> None:
        """Write every field of an existing order."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE orders SET customer_id = ?, status = ?, total = ?, currency = ?, placed_at = ? WHERE id = ?",
            (item.customer_id, item.status, item.total, item.currency, item.placed_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the order with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM orders WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of orders."""
        return self.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]


class OrderItemRepository(BaseRepository):
    """Reads and writes order items in the order_items table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: OrderItem) -> OrderItem:
        """Insert a new order item and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
            (item.order_id, item.product_id, item.quantity, item.unit_price,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[OrderItem]:
        """Return the order item with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM order_items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return OrderItem.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[OrderItem]:
        """Return up to limit order items, ordered by id."""
        rows = self.connection.execute("SELECT * FROM order_items ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [OrderItem.from_dict(dict(row)) for row in rows]

    def update(self, item: OrderItem) -> None:
        """Write every field of an existing order item."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE order_items SET order_id = ?, product_id = ?, quantity = ?, unit_price = ? WHERE id = ?",
            (item.order_id, item.product_id, item.quantity, item.unit_price, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the order item with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM order_items WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of order items."""
        return self.connection.execute("SELECT COUNT(*) FROM order_items").fetchone()[0]


class CustomerRepository(BaseRepository):
    """Reads and writes customers in the customers table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                country TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Customer) -> Customer:
        """Insert a new customer and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO customers (first_name, last_name, email, phone, country) VALUES (?, ?, ?, ?, ?)",
            (item.first_name, item.last_name, item.email, item.phone, item.country,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Customer]:
        """Return the customer with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM customers WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Customer.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Customer]:
        """Return up to limit customers, ordered by id."""
        rows = self.connection.execute("SELECT * FROM customers ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Customer.from_dict(dict(row)) for row in rows]

    def update(self, item: Customer) -> None:
        """Write every field of an existing customer."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE customers SET first_name = ?, last_name = ?, email = ?, phone = ?, country = ? WHERE id = ?",
            (item.first_name, item.last_name, item.email, item.phone, item.country, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the customer with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM customers WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of customers."""
        return self.connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]


class InvoiceRepository(BaseRepository):
    """Reads and writes invoices in the invoices table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                number TEXT NOT NULL,
                amount REAL NOT NULL,
                due_date TEXT NOT NULL,
                paid INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Invoice) -> Invoice:
        """Insert a new invoice and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO invoices (order_id, number, amount, due_date, paid) VALUES (?, ?, ?, ?, ?)",
            (item.order_id, item.number, item.amount, item.due_date.isoformat(), item.paid,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Invoice]:
        """Return the invoice with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM invoices WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Invoice.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Invoice]:
        """Return up to limit invoices, ordered by id."""
        rows = self.connection.execute("SELECT * FROM invoices ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Invoice.from_dict(dict(row)) for row in rows]

    def update(self, item: Invoice) -> None:
        """Write every field of an existing invoice."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE invoices SET order_id = ?, number = ?, amount = ?, due_date = ?, paid = ? WHERE id = ?",
            (item.order_id, item.number, item.amount, item.due_date.isoformat(), item.paid, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the invoice with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM invoices WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of invoices."""
        return self.connection.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]


class PaymentRepository(BaseRepository):
    """Reads and writes payments in the payments table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                amount REAL NOT NULL,
                reference TEXT NOT NULL,
                received_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Payment) -> Payment:
        """Insert a new payment and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO payments (invoice_id, method, amount, reference, received_at) VALUES (?, ?, ?, ?, ?)",
            (item.invoice_id, item.method, item.amount, item.reference, item.received_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Payment]:
        """Return the payment with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM payments WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Payment.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Payment]:
        """Return up to limit payments, ordered by id."""
        rows = self.connection.execute("SELECT * FROM payments ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Payment.from_dict(dict(row)) for row in rows]

    def update(self, item: Payment) -> None:
        """Write every field of an existing payment."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE payments SET invoice_id = ?, method = ?, amount = ?, reference = ?, received_at = ? WHERE id = ?",
            (item.invoice_id, item.method, item.amount, item.reference, item.received_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the payment with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM payments WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of payments."""
        return self.connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0]


class CategoryRepository(BaseRepository):
    """Reads and writes categories in the categories table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                parent_id INTEGER NOT NULL,
                position INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Category) -> Category:
        """Insert a new category and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO categories (name, slug, parent_id, position) VALUES (?, ?, ?, ?)",
            (item.name, item.slug, item.parent_id, item.position,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Category]:
        """Return the category with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM categories WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Category.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Category]:
        """Return up to limit categories, ordered by id."""
        rows = self.connection.execute("SELECT * FROM categories ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Category.from_dict(dict(row)) for row in rows]

    def update(self, item: Category) -> None:
        """Write every field of an existing category."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE categories SET name = ?, slug = ?, parent_id = ?, position = ? WHERE id = ?",
            (item.name, item.slug, item.parent_id, item.position, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the category with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM categories WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of categories."""
        return self.connection.execute("SELECT COUNT(*) FROM categories").fetchone()[0]


class ReviewRepository(BaseRepository):
    """Reads and writes reviews in the reviews table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Review) -> Review:
        """Insert a new review and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO reviews (product_id, user_id, rating, title, body) VALUES (?, ?, ?, ?, ?)",
            (item.product_id, item.user_id, item.rating, item.title, item.body,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Review]:
        """Return the review with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM reviews WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Review.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Review]:
        """Return up to limit reviews, ordered by id."""
        rows = self.connection.execute("SELECT * FROM reviews ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Review.from_dict(dict(row)) for row in rows]

    def update(self, item: Review) -> None:
        """Write every field of an existing review."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE reviews SET product_id = ?, user_id = ?, rating = ?, title = ?, body = ? WHERE id = ?",
            (item.product_id, item.user_id, item.rating, item.title, item.body, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the review with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM reviews WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of reviews."""
        return self.connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]


class AddressRepository(BaseRepository):
    """Reads and writes addresses in the addresses table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS addresses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                street TEXT NOT NULL,
                city TEXT NOT NULL,
                postal_code TEXT NOT NULL,
                country TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Address) -> Address:
        """Insert a new address and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO addresses (customer_id, street, city, postal_code, country) VALUES (?, ?, ?, ?, ?)",
            (item.customer_id, item.street, item.city, item.postal_code, item.country,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Address]:
        """Return the address with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM addresses WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Address.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Address]:
        """Return up to limit addresses, ordered by id."""
        rows = self.connection.execute("SELECT * FROM addresses ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Address.from_dict(dict(row)) for row in rows]

    def update(self, item: Address) -> None:
        """Write every field of an existing address."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE addresses SET customer_id = ?, street = ?, city = ?, postal_code = ?, country = ? WHERE id = ?",
            (item.customer_id, item.street, item.city, item.postal_code, item.country, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the address with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM addresses WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of addresses."""
        return self.connection.execute("SELECT COUNT(*) FROM addresses").fetchone()[0]


class ShipmentRepository(BaseRepository):
    """Reads and writes shipments in the shipments table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS shipments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                carrier TEXT NOT NULL,
                tracking_number TEXT NOT NULL,
                shipped_at TEXT NOT NULL,
                delivered INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Shipment) -> Shipment:
        """Insert a new shipment and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO shipments (order_id, carrier, tracking_number, shipped_at, delivered) VALUES (?, ?, ?, ?, ?)",
            (item.order_id, item.carrier, item.tracking_number, item.shipped_at.isoformat(), item.delivered,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Shipment]:
        """Return the shipment with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM shipments WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Shipment.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Shipment]:
        """Return up to limit shipments, ordered by id."""
        rows = self.connection.execute("SELECT * FROM shipments ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Shipment.from_dict(dict(row)) for row in rows]

    def update(self, item: Shipment) -> None:
        """Write every field of an existing shipment."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE shipments SET order_id = ?, carrier = ?, tracking_number = ?, shipped_at = ?, delivered = ? WHERE id = ?",
            (item.order_id, item.carrier, item.tracking_number, item.shipped_at.isoformat(), item.delivered, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the shipment with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM shipments WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of shipments."""
        return self.connection.execute("SELECT COUNT(*) FROM shipments").fetchone()[0]


class WarehouseRepository(BaseRepository):
    """Reads and writes warehouses in the warehouses table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS warehouses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                capacity INTEGER NOT NULL,
                is_open INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Warehouse) -> Warehouse:
        """Insert a new warehouse and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO warehouses (name, city, capacity, is_open) VALUES (?, ?, ?, ?)",
            (item.name, item.city, item.capacity, item.is_open,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Warehouse]:
        """Return the warehouse with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM warehouses WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Warehouse.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Warehouse]:
        """Return up to limit warehouses, ordered by id."""
        rows = self.connection.execute("SELECT * FROM warehouses ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Warehouse.from_dict(dict(row)) for row in rows]

    def update(self, item: Warehouse) -> None:
        """Write every field of an existing warehouse."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE warehouses SET name = ?, city = ?, capacity = ?, is_open = ? WHERE id = ?",
            (item.name, item.city, item.capacity, item.is_open, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the warehouse with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM warehouses WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of warehouses."""
        return self.connection.execute("SELECT COUNT(*) FROM warehouses").fetchone()[0]


class SupplierRepository(BaseRepository):
    """Reads and writes suppliers in the suppliers table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS suppliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact_email TEXT NOT NULL,
                phone TEXT NOT NULL,
                country TEXT NOT NULL,
                rating INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Supplier) -> Supplier:
        """Insert a new supplier and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO suppliers (name, contact_email, phone, country, rating) VALUES (?, ?, ?, ?, ?)",
            (item.name, item.contact_email, item.phone, item.country, item.rating,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Supplier]:
        """Return the supplier with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM suppliers WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Supplier.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Supplier]:
        """Return up to limit suppliers, ordered by id."""
        rows = self.connection.execute("SELECT * FROM suppliers ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Supplier.from_dict(dict(row)) for row in rows]

    def update(self, item: Supplier) -> None:
        """Write every field of an existing supplier."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE suppliers SET name = ?, contact_email = ?, phone = ?, country = ?, rating = ? WHERE id = ?",
            (item.name, item.contact_email, item.phone, item.country, item.rating, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the supplier with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM suppliers WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of suppliers."""
        return self.connection.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]


class CouponRepository(BaseRepository):
    """Reads and writes coupons in the coupons table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS coupons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                discount_percent INTEGER NOT NULL,
                valid_until TEXT NOT NULL,
                max_uses INTEGER NOT NULL,
                is_active INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Coupon) -> Coupon:
        """Insert a new coupon and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO coupons (code, discount_percent, valid_until, max_uses, is_active) VALUES (?, ?, ?, ?, ?)",
            (item.code, item.discount_percent, item.valid_until.isoformat(), item.max_uses, item.is_active,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Coupon]:
        """Return the coupon with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM coupons WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Coupon.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Coupon]:
        """Return up to limit coupons, ordered by id."""
        rows = self.connection.execute("SELECT * FROM coupons ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Coupon.from_dict(dict(row)) for row in rows]

    def update(self, item: Coupon) -> None:
        """Write every field of an existing coupon."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE coupons SET code = ?, discount_percent = ?, valid_until = ?, max_uses = ?, is_active = ? WHERE id = ?",
            (item.code, item.discount_percent, item.valid_until.isoformat(), item.max_uses, item.is_active, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the coupon with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM coupons WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of coupons."""
        return self.connection.execute("SELECT COUNT(*) FROM coupons").fetchone()[0]


class EmployeeRepository(BaseRepository):
    """Reads and writes employees in the employees table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Employee) -> Employee:
        """Insert a new employee and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO employees (first_name, last_name, email, department, salary) VALUES (?, ?, ?, ?, ?)",
            (item.first_name, item.last_name, item.email, item.department, item.salary,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Employee]:
        """Return the employee with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM employees WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Employee.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Employee]:
        """Return up to limit employees, ordered by id."""
        rows = self.connection.execute("SELECT * FROM employees ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Employee.from_dict(dict(row)) for row in rows]

    def update(self, item: Employee) -> None:
        """Write every field of an existing employee."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE employees SET first_name = ?, last_name = ?, email = ?, department = ?, salary = ? WHERE id = ?",
            (item.first_name, item.last_name, item.email, item.department, item.salary, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the employee with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM employees WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of employees."""
        return self.connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0]


class DepartmentRepository(BaseRepository):
    """Reads and writes departments in the departments table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT NOT NULL,
                manager_id INTEGER NOT NULL,
                budget REAL NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Department) -> Department:
        """Insert a new department and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO departments (name, code, manager_id, budget) VALUES (?, ?, ?, ?)",
            (item.name, item.code, item.manager_id, item.budget,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Department]:
        """Return the department with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM departments WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Department.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Department]:
        """Return up to limit departments, ordered by id."""
        rows = self.connection.execute("SELECT * FROM departments ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Department.from_dict(dict(row)) for row in rows]

    def update(self, item: Department) -> None:
        """Write every field of an existing department."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE departments SET name = ?, code = ?, manager_id = ?, budget = ? WHERE id = ?",
            (item.name, item.code, item.manager_id, item.budget, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the department with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM departments WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of departments."""
        return self.connection.execute("SELECT COUNT(*) FROM departments").fetchone()[0]


class ProjectRepository(BaseRepository):
    """Reads and writes projects in the projects table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                budget REAL NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Project) -> Project:
        """Insert a new project and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO projects (name, description, start_date, end_date, budget) VALUES (?, ?, ?, ?, ?)",
            (item.name, item.description, item.start_date.isoformat(), item.end_date.isoformat(), item.budget,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Project]:
        """Return the project with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM projects WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Project.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Project]:
        """Return up to limit projects, ordered by id."""
        rows = self.connection.execute("SELECT * FROM projects ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Project.from_dict(dict(row)) for row in rows]

    def update(self, item: Project) -> None:
        """Write every field of an existing project."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE projects SET name = ?, description = ?, start_date = ?, end_date = ?, budget = ? WHERE id = ?",
            (item.name, item.description, item.start_date.isoformat(), item.end_date.isoformat(), item.budget, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the project with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM projects WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of projects."""
        return self.connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]


class TaskRepository(BaseRepository):
    """Reads and writes tasks in the tasks table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL,
                due_date TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Task) -> Task:
        """Insert a new task and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO tasks (project_id, title, status, priority, due_date) VALUES (?, ?, ?, ?, ?)",
            (item.project_id, item.title, item.status, item.priority, item.due_date.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Task]:
        """Return the task with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM tasks WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Task.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Task]:
        """Return up to limit tasks, ordered by id."""
        rows = self.connection.execute("SELECT * FROM tasks ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Task.from_dict(dict(row)) for row in rows]

    def update(self, item: Task) -> None:
        """Write every field of an existing task."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE tasks SET project_id = ?, title = ?, status = ?, priority = ?, due_date = ? WHERE id = ?",
            (item.project_id, item.title, item.status, item.priority, item.due_date.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the task with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM tasks WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of tasks."""
        return self.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]


class CommentRepository(BaseRepository):
    """Reads and writes comments in the comments table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Comment) -> Comment:
        """Insert a new comment and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO comments (task_id, author_id, body, created_at) VALUES (?, ?, ?, ?)",
            (item.task_id, item.author_id, item.body, item.created_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Comment]:
        """Return the comment with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM comments WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Comment.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Comment]:
        """Return up to limit comments, ordered by id."""
        rows = self.connection.execute("SELECT * FROM comments ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Comment.from_dict(dict(row)) for row in rows]

    def update(self, item: Comment) -> None:
        """Write every field of an existing comment."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE comments SET task_id = ?, author_id = ?, body = ?, created_at = ? WHERE id = ?",
            (item.task_id, item.author_id, item.body, item.created_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the comment with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM comments WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of comments."""
        return self.connection.execute("SELECT COUNT(*) FROM comments").fetchone()[0]


class TagRepository(BaseRepository):
    """Reads and writes tags in the tags table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                color TEXT NOT NULL,
                usage_count INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Tag) -> Tag:
        """Insert a new tag and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO tags (name, color, usage_count) VALUES (?, ?, ?)",
            (item.name, item.color, item.usage_count,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Tag]:
        """Return the tag with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM tags WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Tag.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Tag]:
        """Return up to limit tags, ordered by id."""
        rows = self.connection.execute("SELECT * FROM tags ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Tag.from_dict(dict(row)) for row in rows]

    def update(self, item: Tag) -> None:
        """Write every field of an existing tag."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE tags SET name = ?, color = ?, usage_count = ? WHERE id = ?",
            (item.name, item.color, item.usage_count, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the tag with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM tags WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of tags."""
        return self.connection.execute("SELECT COUNT(*) FROM tags").fetchone()[0]


class ArticleRepository(BaseRepository):
    """Reads and writes articles in the articles table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                slug TEXT NOT NULL,
                body TEXT NOT NULL,
                author_id INTEGER NOT NULL,
                published INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Article) -> Article:
        """Insert a new article and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO articles (title, slug, body, author_id, published) VALUES (?, ?, ?, ?, ?)",
            (item.title, item.slug, item.body, item.author_id, item.published,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Article]:
        """Return the article with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM articles WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Article.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Article]:
        """Return up to limit articles, ordered by id."""
        rows = self.connection.execute("SELECT * FROM articles ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Article.from_dict(dict(row)) for row in rows]

    def update(self, item: Article) -> None:
        """Write every field of an existing article."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE articles SET title = ?, slug = ?, body = ?, author_id = ?, published = ? WHERE id = ?",
            (item.title, item.slug, item.body, item.author_id, item.published, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the article with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM articles WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of articles."""
        return self.connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]


class AuthorRepository(BaseRepository):
    """Reads and writes authors in the authors table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                bio TEXT NOT NULL,
                email TEXT NOT NULL,
                website TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Author) -> Author:
        """Insert a new author and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO authors (name, bio, email, website) VALUES (?, ?, ?, ?)",
            (item.name, item.bio, item.email, item.website,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Author]:
        """Return the author with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM authors WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Author.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Author]:
        """Return up to limit authors, ordered by id."""
        rows = self.connection.execute("SELECT * FROM authors ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Author.from_dict(dict(row)) for row in rows]

    def update(self, item: Author) -> None:
        """Write every field of an existing author."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE authors SET name = ?, bio = ?, email = ?, website = ? WHERE id = ?",
            (item.name, item.bio, item.email, item.website, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the author with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM authors WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of authors."""
        return self.connection.execute("SELECT COUNT(*) FROM authors").fetchone()[0]


class BookRepository(BaseRepository):
    """Reads and writes books in the books table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                isbn TEXT NOT NULL,
                author_id INTEGER NOT NULL,
                year INTEGER NOT NULL,
                pages INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Book) -> Book:
        """Insert a new book and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO books (title, isbn, author_id, year, pages) VALUES (?, ?, ?, ?, ?)",
            (item.title, item.isbn, item.author_id, item.year, item.pages,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Book]:
        """Return the book with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM books WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Book.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Book]:
        """Return up to limit books, ordered by id."""
        rows = self.connection.execute("SELECT * FROM books ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Book.from_dict(dict(row)) for row in rows]

    def update(self, item: Book) -> None:
        """Write every field of an existing book."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE books SET title = ?, isbn = ?, author_id = ?, year = ?, pages = ? WHERE id = ?",
            (item.title, item.isbn, item.author_id, item.year, item.pages, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the book with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM books WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of books."""
        return self.connection.execute("SELECT COUNT(*) FROM books").fetchone()[0]


class LibraryRepository(BaseRepository):
    """Reads and writes libraries in the libraries table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS libraries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                opening_hour INTEGER NOT NULL,
                closing_hour INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Library) -> Library:
        """Insert a new library and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO libraries (name, city, opening_hour, closing_hour) VALUES (?, ?, ?, ?)",
            (item.name, item.city, item.opening_hour, item.closing_hour,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Library]:
        """Return the library with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM libraries WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Library.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Library]:
        """Return up to limit libraries, ordered by id."""
        rows = self.connection.execute("SELECT * FROM libraries ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Library.from_dict(dict(row)) for row in rows]

    def update(self, item: Library) -> None:
        """Write every field of an existing library."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE libraries SET name = ?, city = ?, opening_hour = ?, closing_hour = ? WHERE id = ?",
            (item.name, item.city, item.opening_hour, item.closing_hour, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the library with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM libraries WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of libraries."""
        return self.connection.execute("SELECT COUNT(*) FROM libraries").fetchone()[0]


class LoanRepository(BaseRepository):
    """Reads and writes loans in the loans table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS loans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                member_id INTEGER NOT NULL,
                borrowed_at TEXT NOT NULL,
                due_date TEXT NOT NULL,
                returned INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Loan) -> Loan:
        """Insert a new loan and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO loans (book_id, member_id, borrowed_at, due_date, returned) VALUES (?, ?, ?, ?, ?)",
            (item.book_id, item.member_id, item.borrowed_at.isoformat(), item.due_date.isoformat(), item.returned,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Loan]:
        """Return the loan with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM loans WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Loan.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Loan]:
        """Return up to limit loans, ordered by id."""
        rows = self.connection.execute("SELECT * FROM loans ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Loan.from_dict(dict(row)) for row in rows]

    def update(self, item: Loan) -> None:
        """Write every field of an existing loan."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE loans SET book_id = ?, member_id = ?, borrowed_at = ?, due_date = ?, returned = ? WHERE id = ?",
            (item.book_id, item.member_id, item.borrowed_at.isoformat(), item.due_date.isoformat(), item.returned, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the loan with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM loans WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of loans."""
        return self.connection.execute("SELECT COUNT(*) FROM loans").fetchone()[0]


class MemberRepository(BaseRepository):
    """Reads and writes members in the members table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL,
                joined_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Member) -> Member:
        """Insert a new member and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO members (first_name, last_name, email, joined_at) VALUES (?, ?, ?, ?)",
            (item.first_name, item.last_name, item.email, item.joined_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Member]:
        """Return the member with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM members WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Member.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Member]:
        """Return up to limit members, ordered by id."""
        rows = self.connection.execute("SELECT * FROM members ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Member.from_dict(dict(row)) for row in rows]

    def update(self, item: Member) -> None:
        """Write every field of an existing member."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE members SET first_name = ?, last_name = ?, email = ?, joined_at = ? WHERE id = ?",
            (item.first_name, item.last_name, item.email, item.joined_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the member with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM members WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of members."""
        return self.connection.execute("SELECT COUNT(*) FROM members").fetchone()[0]


class EventRepository(BaseRepository):
    """Reads and writes events in the events table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                location TEXT NOT NULL,
                starts_at TEXT NOT NULL,
                ends_at TEXT NOT NULL,
                capacity INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Event) -> Event:
        """Insert a new event and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO events (title, location, starts_at, ends_at, capacity) VALUES (?, ?, ?, ?, ?)",
            (item.title, item.location, item.starts_at.isoformat(), item.ends_at.isoformat(), item.capacity,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Event]:
        """Return the event with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM events WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Event.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Event]:
        """Return up to limit events, ordered by id."""
        rows = self.connection.execute("SELECT * FROM events ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Event.from_dict(dict(row)) for row in rows]

    def update(self, item: Event) -> None:
        """Write every field of an existing event."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE events SET title = ?, location = ?, starts_at = ?, ends_at = ?, capacity = ? WHERE id = ?",
            (item.title, item.location, item.starts_at.isoformat(), item.ends_at.isoformat(), item.capacity, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the event with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM events WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of events."""
        return self.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]


class TicketRepository(BaseRepository):
    """Reads and writes tickets in the tickets table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                holder_name TEXT NOT NULL,
                seat TEXT NOT NULL,
                price REAL NOT NULL,
                checked_in INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Ticket) -> Ticket:
        """Insert a new ticket and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO tickets (event_id, holder_name, seat, price, checked_in) VALUES (?, ?, ?, ?, ?)",
            (item.event_id, item.holder_name, item.seat, item.price, item.checked_in,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Ticket]:
        """Return the ticket with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM tickets WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Ticket.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Ticket]:
        """Return up to limit tickets, ordered by id."""
        rows = self.connection.execute("SELECT * FROM tickets ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Ticket.from_dict(dict(row)) for row in rows]

    def update(self, item: Ticket) -> None:
        """Write every field of an existing ticket."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE tickets SET event_id = ?, holder_name = ?, seat = ?, price = ?, checked_in = ? WHERE id = ?",
            (item.event_id, item.holder_name, item.seat, item.price, item.checked_in, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the ticket with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM tickets WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of tickets."""
        return self.connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


class VenueRepository(BaseRepository):
    """Reads and writes venues in the venues table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS venues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT NOT NULL,
                city TEXT NOT NULL,
                capacity INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Venue) -> Venue:
        """Insert a new venue and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO venues (name, address, city, capacity) VALUES (?, ?, ?, ?)",
            (item.name, item.address, item.city, item.capacity,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Venue]:
        """Return the venue with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM venues WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Venue.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Venue]:
        """Return up to limit venues, ordered by id."""
        rows = self.connection.execute("SELECT * FROM venues ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Venue.from_dict(dict(row)) for row in rows]

    def update(self, item: Venue) -> None:
        """Write every field of an existing venue."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE venues SET name = ?, address = ?, city = ?, capacity = ? WHERE id = ?",
            (item.name, item.address, item.city, item.capacity, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the venue with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM venues WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of venues."""
        return self.connection.execute("SELECT COUNT(*) FROM venues").fetchone()[0]


class VehicleRepository(BaseRepository):
    """Reads and writes vehicles in the vehicles table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                year INTEGER NOT NULL,
                mileage INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Vehicle) -> Vehicle:
        """Insert a new vehicle and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO vehicles (plate, make, model, year, mileage) VALUES (?, ?, ?, ?, ?)",
            (item.plate, item.make, item.model, item.year, item.mileage,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Vehicle]:
        """Return the vehicle with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM vehicles WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Vehicle.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Vehicle]:
        """Return up to limit vehicles, ordered by id."""
        rows = self.connection.execute("SELECT * FROM vehicles ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Vehicle.from_dict(dict(row)) for row in rows]

    def update(self, item: Vehicle) -> None:
        """Write every field of an existing vehicle."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE vehicles SET plate = ?, make = ?, model = ?, year = ?, mileage = ? WHERE id = ?",
            (item.plate, item.make, item.model, item.year, item.mileage, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the vehicle with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM vehicles WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of vehicles."""
        return self.connection.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]


class DriverRepository(BaseRepository):
    """Reads and writes drivers in the drivers table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS drivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                license_number TEXT NOT NULL,
                rating REAL NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Driver) -> Driver:
        """Insert a new driver and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO drivers (first_name, last_name, license_number, rating) VALUES (?, ?, ?, ?)",
            (item.first_name, item.last_name, item.license_number, item.rating,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Driver]:
        """Return the driver with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM drivers WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Driver.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Driver]:
        """Return up to limit drivers, ordered by id."""
        rows = self.connection.execute("SELECT * FROM drivers ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Driver.from_dict(dict(row)) for row in rows]

    def update(self, item: Driver) -> None:
        """Write every field of an existing driver."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE drivers SET first_name = ?, last_name = ?, license_number = ?, rating = ? WHERE id = ?",
            (item.first_name, item.last_name, item.license_number, item.rating, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the driver with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM drivers WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of drivers."""
        return self.connection.execute("SELECT COUNT(*) FROM drivers").fetchone()[0]


class TripRepository(BaseRepository):
    """Reads and writes trips in the trips table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER NOT NULL,
                driver_id INTEGER NOT NULL,
                distance_km REAL NOT NULL,
                started_at TEXT NOT NULL,
                fare REAL NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Trip) -> Trip:
        """Insert a new trip and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO trips (vehicle_id, driver_id, distance_km, started_at, fare) VALUES (?, ?, ?, ?, ?)",
            (item.vehicle_id, item.driver_id, item.distance_km, item.started_at.isoformat(), item.fare,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Trip]:
        """Return the trip with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM trips WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Trip.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Trip]:
        """Return up to limit trips, ordered by id."""
        rows = self.connection.execute("SELECT * FROM trips ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Trip.from_dict(dict(row)) for row in rows]

    def update(self, item: Trip) -> None:
        """Write every field of an existing trip."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE trips SET vehicle_id = ?, driver_id = ?, distance_km = ?, started_at = ?, fare = ? WHERE id = ?",
            (item.vehicle_id, item.driver_id, item.distance_km, item.started_at.isoformat(), item.fare, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the trip with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM trips WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of trips."""
        return self.connection.execute("SELECT COUNT(*) FROM trips").fetchone()[0]


class PatientRepository(BaseRepository):
    """Reads and writes patients in the patients table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                birth_date TEXT NOT NULL,
                phone TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Patient) -> Patient:
        """Insert a new patient and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO patients (first_name, last_name, birth_date, phone) VALUES (?, ?, ?, ?)",
            (item.first_name, item.last_name, item.birth_date.isoformat(), item.phone,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Patient]:
        """Return the patient with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM patients WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Patient.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Patient]:
        """Return up to limit patients, ordered by id."""
        rows = self.connection.execute("SELECT * FROM patients ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Patient.from_dict(dict(row)) for row in rows]

    def update(self, item: Patient) -> None:
        """Write every field of an existing patient."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE patients SET first_name = ?, last_name = ?, birth_date = ?, phone = ? WHERE id = ?",
            (item.first_name, item.last_name, item.birth_date.isoformat(), item.phone, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the patient with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM patients WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of patients."""
        return self.connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]


class DoctorRepository(BaseRepository):
    """Reads and writes doctors in the doctors table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                room TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Doctor) -> Doctor:
        """Insert a new doctor and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO doctors (first_name, last_name, specialty, room) VALUES (?, ?, ?, ?)",
            (item.first_name, item.last_name, item.specialty, item.room,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Doctor]:
        """Return the doctor with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM doctors WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Doctor.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Doctor]:
        """Return up to limit doctors, ordered by id."""
        rows = self.connection.execute("SELECT * FROM doctors ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Doctor.from_dict(dict(row)) for row in rows]

    def update(self, item: Doctor) -> None:
        """Write every field of an existing doctor."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE doctors SET first_name = ?, last_name = ?, specialty = ?, room = ? WHERE id = ?",
            (item.first_name, item.last_name, item.specialty, item.room, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the doctor with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM doctors WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of doctors."""
        return self.connection.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]


class AppointmentRepository(BaseRepository):
    """Reads and writes appointments in the appointments table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                scheduled_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                confirmed INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Appointment) -> Appointment:
        """Insert a new appointment and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO appointments (patient_id, doctor_id, scheduled_at, reason, confirmed) VALUES (?, ?, ?, ?, ?)",
            (item.patient_id, item.doctor_id, item.scheduled_at.isoformat(), item.reason, item.confirmed,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Appointment]:
        """Return the appointment with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM appointments WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Appointment.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Appointment]:
        """Return up to limit appointments, ordered by id."""
        rows = self.connection.execute("SELECT * FROM appointments ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Appointment.from_dict(dict(row)) for row in rows]

    def update(self, item: Appointment) -> None:
        """Write every field of an existing appointment."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE appointments SET patient_id = ?, doctor_id = ?, scheduled_at = ?, reason = ?, confirmed = ? WHERE id = ?",
            (item.patient_id, item.doctor_id, item.scheduled_at.isoformat(), item.reason, item.confirmed, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the appointment with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM appointments WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of appointments."""
        return self.connection.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]


class CourseRepository(BaseRepository):
    """Reads and writes courses in the courses table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                code TEXT NOT NULL,
                credits INTEGER NOT NULL,
                teacher_id INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Course) -> Course:
        """Insert a new course and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO courses (title, code, credits, teacher_id) VALUES (?, ?, ?, ?)",
            (item.title, item.code, item.credits, item.teacher_id,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Course]:
        """Return the course with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM courses WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Course.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Course]:
        """Return up to limit courses, ordered by id."""
        rows = self.connection.execute("SELECT * FROM courses ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Course.from_dict(dict(row)) for row in rows]

    def update(self, item: Course) -> None:
        """Write every field of an existing course."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE courses SET title = ?, code = ?, credits = ?, teacher_id = ? WHERE id = ?",
            (item.title, item.code, item.credits, item.teacher_id, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the course with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM courses WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of courses."""
        return self.connection.execute("SELECT COUNT(*) FROM courses").fetchone()[0]


class StudentRepository(BaseRepository):
    """Reads and writes students in the students table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL,
                enrolled_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Student) -> Student:
        """Insert a new student and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO students (first_name, last_name, email, enrolled_at) VALUES (?, ?, ?, ?)",
            (item.first_name, item.last_name, item.email, item.enrolled_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Student]:
        """Return the student with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM students WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Student.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Student]:
        """Return up to limit students, ordered by id."""
        rows = self.connection.execute("SELECT * FROM students ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Student.from_dict(dict(row)) for row in rows]

    def update(self, item: Student) -> None:
        """Write every field of an existing student."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE students SET first_name = ?, last_name = ?, email = ?, enrolled_at = ? WHERE id = ?",
            (item.first_name, item.last_name, item.email, item.enrolled_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the student with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM students WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of students."""
        return self.connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]


class EnrollmentRepository(BaseRepository):
    """Reads and writes enrollments in the enrollments table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                grade REAL NOT NULL,
                completed INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Enrollment) -> Enrollment:
        """Insert a new enrollment and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO enrollments (student_id, course_id, grade, completed) VALUES (?, ?, ?, ?)",
            (item.student_id, item.course_id, item.grade, item.completed,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Enrollment]:
        """Return the enrollment with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM enrollments WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Enrollment.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Enrollment]:
        """Return up to limit enrollments, ordered by id."""
        rows = self.connection.execute("SELECT * FROM enrollments ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Enrollment.from_dict(dict(row)) for row in rows]

    def update(self, item: Enrollment) -> None:
        """Write every field of an existing enrollment."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE enrollments SET student_id = ?, course_id = ?, grade = ?, completed = ? WHERE id = ?",
            (item.student_id, item.course_id, item.grade, item.completed, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the enrollment with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM enrollments WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of enrollments."""
        return self.connection.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]


class RoomRepository(BaseRepository):
    """Reads and writes rooms in the rooms table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number TEXT NOT NULL,
                floor INTEGER NOT NULL,
                beds INTEGER NOT NULL,
                price_per_night REAL NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Room) -> Room:
        """Insert a new room and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO rooms (number, floor, beds, price_per_night) VALUES (?, ?, ?, ?)",
            (item.number, item.floor, item.beds, item.price_per_night,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Room]:
        """Return the room with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM rooms WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Room.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Room]:
        """Return up to limit rooms, ordered by id."""
        rows = self.connection.execute("SELECT * FROM rooms ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Room.from_dict(dict(row)) for row in rows]

    def update(self, item: Room) -> None:
        """Write every field of an existing room."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE rooms SET number = ?, floor = ?, beds = ?, price_per_night = ? WHERE id = ?",
            (item.number, item.floor, item.beds, item.price_per_night, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the room with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM rooms WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of rooms."""
        return self.connection.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]


class BookingRepository(BaseRepository):
    """Reads and writes bookings in the bookings table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id INTEGER NOT NULL,
                guest_name TEXT NOT NULL,
                check_in TEXT NOT NULL,
                check_out TEXT NOT NULL,
                paid INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Booking) -> Booking:
        """Insert a new booking and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO bookings (room_id, guest_name, check_in, check_out, paid) VALUES (?, ?, ?, ?, ?)",
            (item.room_id, item.guest_name, item.check_in.isoformat(), item.check_out.isoformat(), item.paid,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Booking]:
        """Return the booking with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM bookings WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Booking.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Booking]:
        """Return up to limit bookings, ordered by id."""
        rows = self.connection.execute("SELECT * FROM bookings ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Booking.from_dict(dict(row)) for row in rows]

    def update(self, item: Booking) -> None:
        """Write every field of an existing booking."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE bookings SET room_id = ?, guest_name = ?, check_in = ?, check_out = ?, paid = ? WHERE id = ?",
            (item.room_id, item.guest_name, item.check_in.isoformat(), item.check_out.isoformat(), item.paid, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the booking with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM bookings WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of bookings."""
        return self.connection.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]


class SubscriptionRepository(BaseRepository):
    """Reads and writes subscriptions in the subscriptions table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan TEXT NOT NULL,
                price REAL NOT NULL,
                renews_at TEXT NOT NULL,
                is_active INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Subscription) -> Subscription:
        """Insert a new subscription and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO subscriptions (user_id, plan, price, renews_at, is_active) VALUES (?, ?, ?, ?, ?)",
            (item.user_id, item.plan, item.price, item.renews_at.isoformat(), item.is_active,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Subscription]:
        """Return the subscription with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM subscriptions WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Subscription.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Subscription]:
        """Return up to limit subscriptions, ordered by id."""
        rows = self.connection.execute("SELECT * FROM subscriptions ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Subscription.from_dict(dict(row)) for row in rows]

    def update(self, item: Subscription) -> None:
        """Write every field of an existing subscription."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE subscriptions SET user_id = ?, plan = ?, price = ?, renews_at = ?, is_active = ? WHERE id = ?",
            (item.user_id, item.plan, item.price, item.renews_at.isoformat(), item.is_active, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the subscription with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM subscriptions WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of subscriptions."""
        return self.connection.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]


class NotificationRepository(BaseRepository):
    """Reads and writes notifications in the notifications table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                read INTEGER NOT NULL,
                sent_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Notification) -> Notification:
        """Insert a new notification and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO notifications (user_id, title, message, read, sent_at) VALUES (?, ?, ?, ?, ?)",
            (item.user_id, item.title, item.message, item.read, item.sent_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Notification]:
        """Return the notification with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM notifications WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Notification.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Notification]:
        """Return up to limit notifications, ordered by id."""
        rows = self.connection.execute("SELECT * FROM notifications ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Notification.from_dict(dict(row)) for row in rows]

    def update(self, item: Notification) -> None:
        """Write every field of an existing notification."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE notifications SET user_id = ?, title = ?, message = ?, read = ?, sent_at = ? WHERE id = ?",
            (item.user_id, item.title, item.message, item.read, item.sent_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the notification with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM notifications WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of notifications."""
        return self.connection.execute("SELECT COUNT(*) FROM notifications").fetchone()[0]


class SessionRepository(BaseRepository):
    """Reads and writes sessions in the sessions table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Session) -> Session:
        """Insert a new session and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO sessions (user_id, token, expires_at, ip_address) VALUES (?, ?, ?, ?)",
            (item.user_id, item.token, item.expires_at.isoformat(), item.ip_address,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Session]:
        """Return the session with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM sessions WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Session.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Session]:
        """Return up to limit sessions, ordered by id."""
        rows = self.connection.execute("SELECT * FROM sessions ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Session.from_dict(dict(row)) for row in rows]

    def update(self, item: Session) -> None:
        """Write every field of an existing session."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE sessions SET user_id = ?, token = ?, expires_at = ?, ip_address = ? WHERE id = ?",
            (item.user_id, item.token, item.expires_at.isoformat(), item.ip_address, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the session with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM sessions WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of sessions."""
        return self.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]


class AuditLogRepository(BaseRepository):
    """Reads and writes audit logs in the audit_logs table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: AuditLog) -> AuditLog:
        """Insert a new audit log and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO audit_logs (actor_id, action, target, created_at) VALUES (?, ?, ?, ?)",
            (item.actor_id, item.action, item.target, item.created_at.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[AuditLog]:
        """Return the audit log with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM audit_logs WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return AuditLog.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[AuditLog]:
        """Return up to limit audit logs, ordered by id."""
        rows = self.connection.execute("SELECT * FROM audit_logs ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [AuditLog.from_dict(dict(row)) for row in rows]

    def update(self, item: AuditLog) -> None:
        """Write every field of an existing audit log."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE audit_logs SET actor_id = ?, action = ?, target = ?, created_at = ? WHERE id = ?",
            (item.actor_id, item.action, item.target, item.created_at.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the audit log with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM audit_logs WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of audit logs."""
        return self.connection.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]


class SettingRepository(BaseRepository):
    """Reads and writes settings in the settings table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                description TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Setting) -> Setting:
        """Insert a new setting and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO settings (key, value, description) VALUES (?, ?, ?)",
            (item.key, item.value, item.description,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Setting]:
        """Return the setting with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM settings WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Setting.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Setting]:
        """Return up to limit settings, ordered by id."""
        rows = self.connection.execute("SELECT * FROM settings ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Setting.from_dict(dict(row)) for row in rows]

    def update(self, item: Setting) -> None:
        """Write every field of an existing setting."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE settings SET key = ?, value = ?, description = ? WHERE id = ?",
            (item.key, item.value, item.description, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the setting with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM settings WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of settings."""
        return self.connection.execute("SELECT COUNT(*) FROM settings").fetchone()[0]


class RecipeRepository(BaseRepository):
    """Reads and writes recipes in the recipes table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                servings INTEGER NOT NULL,
                prep_minutes INTEGER NOT NULL,
                cook_minutes INTEGER NOT NULL,
                vegetarian INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Recipe) -> Recipe:
        """Insert a new recipe and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO recipes (title, servings, prep_minutes, cook_minutes, vegetarian) VALUES (?, ?, ?, ?, ?)",
            (item.title, item.servings, item.prep_minutes, item.cook_minutes, item.vegetarian,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Recipe]:
        """Return the recipe with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM recipes WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Recipe.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Recipe]:
        """Return up to limit recipes, ordered by id."""
        rows = self.connection.execute("SELECT * FROM recipes ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Recipe.from_dict(dict(row)) for row in rows]

    def update(self, item: Recipe) -> None:
        """Write every field of an existing recipe."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE recipes SET title = ?, servings = ?, prep_minutes = ?, cook_minutes = ?, vegetarian = ? WHERE id = ?",
            (item.title, item.servings, item.prep_minutes, item.cook_minutes, item.vegetarian, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the recipe with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM recipes WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of recipes."""
        return self.connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]


class IngredientRepository(BaseRepository):
    """Reads and writes ingredients in the ingredients table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                amount REAL NOT NULL,
                unit TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Ingredient) -> Ingredient:
        """Insert a new ingredient and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO ingredients (recipe_id, name, amount, unit) VALUES (?, ?, ?, ?)",
            (item.recipe_id, item.name, item.amount, item.unit,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Ingredient]:
        """Return the ingredient with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM ingredients WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Ingredient.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Ingredient]:
        """Return up to limit ingredients, ordered by id."""
        rows = self.connection.execute("SELECT * FROM ingredients ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Ingredient.from_dict(dict(row)) for row in rows]

    def update(self, item: Ingredient) -> None:
        """Write every field of an existing ingredient."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE ingredients SET recipe_id = ?, name = ?, amount = ?, unit = ? WHERE id = ?",
            (item.recipe_id, item.name, item.amount, item.unit, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the ingredient with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM ingredients WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of ingredients."""
        return self.connection.execute("SELECT COUNT(*) FROM ingredients").fetchone()[0]


class PlaylistRepository(BaseRepository):
    """Reads and writes playlists in the playlists table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                owner_id INTEGER NOT NULL,
                is_public INTEGER NOT NULL,
                track_count INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Playlist) -> Playlist:
        """Insert a new playlist and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO playlists (name, owner_id, is_public, track_count) VALUES (?, ?, ?, ?)",
            (item.name, item.owner_id, item.is_public, item.track_count,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Playlist]:
        """Return the playlist with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM playlists WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Playlist.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Playlist]:
        """Return up to limit playlists, ordered by id."""
        rows = self.connection.execute("SELECT * FROM playlists ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Playlist.from_dict(dict(row)) for row in rows]

    def update(self, item: Playlist) -> None:
        """Write every field of an existing playlist."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE playlists SET name = ?, owner_id = ?, is_public = ?, track_count = ? WHERE id = ?",
            (item.name, item.owner_id, item.is_public, item.track_count, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the playlist with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM playlists WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of playlists."""
        return self.connection.execute("SELECT COUNT(*) FROM playlists").fetchone()[0]


class TrackRepository(BaseRepository):
    """Reads and writes tracks in the tracks table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                album TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Track) -> Track:
        """Insert a new track and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO tracks (title, artist, album, duration_seconds) VALUES (?, ?, ?, ?)",
            (item.title, item.artist, item.album, item.duration_seconds,),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Track]:
        """Return the track with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM tracks WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Track.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Track]:
        """Return up to limit tracks, ordered by id."""
        rows = self.connection.execute("SELECT * FROM tracks ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Track.from_dict(dict(row)) for row in rows]

    def update(self, item: Track) -> None:
        """Write every field of an existing track."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE tracks SET title = ?, artist = ?, album = ?, duration_seconds = ? WHERE id = ?",
            (item.title, item.artist, item.album, item.duration_seconds, item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the track with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM tracks WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of tracks."""
        return self.connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]


class DeviceRepository(BaseRepository):
    """Reads and writes devices in the devices table."""

    def create_table(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                serial_number TEXT NOT NULL,
                firmware TEXT NOT NULL,
                online INTEGER NOT NULL,
                last_seen TEXT NOT NULL
            )"""
        )
        self.commit()

    def add(self, item: Device) -> Device:
        """Insert a new device and return it with its id set."""
        cursor = self.connection.execute(
            "INSERT INTO devices (name, serial_number, firmware, online, last_seen) VALUES (?, ?, ?, ?, ?)",
            (item.name, item.serial_number, item.firmware, item.online, item.last_seen.isoformat(),),
        )
        self.commit()
        item.id = cursor.lastrowid
        return item

    def get(self, item_id: int) -> Optional[Device]:
        """Return the device with this id, or None when there is none."""
        row = self.connection.execute("SELECT * FROM devices WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return Device.from_dict(dict(row))

    def list(self, limit: int = 100, offset: int = 0) -> list[Device]:
        """Return up to limit devices, ordered by id."""
        rows = self.connection.execute("SELECT * FROM devices ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [Device.from_dict(dict(row)) for row in rows]

    def update(self, item: Device) -> None:
        """Write every field of an existing device."""
        if item.id is None:
            raise ValueError("cannot update a record without an id")
        self.connection.execute(
            "UPDATE devices SET name = ?, serial_number = ?, firmware = ?, online = ?, last_seen = ? WHERE id = ?",
            (item.name, item.serial_number, item.firmware, item.online, item.last_seen.isoformat(), item.id),
        )
        self.commit()

    def delete(self, item_id: int) -> bool:
        """Delete the device with this id; return True when a row was removed."""
        cursor = self.connection.execute("DELETE FROM devices WHERE id = ?", (item_id,))
        self.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of devices."""
        return self.connection.execute("SELECT COUNT(*) FROM devices").fetchone()[0]

