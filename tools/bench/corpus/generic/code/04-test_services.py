"""Tests for the service layer, run with pytest."""
import sqlite3

import pytest

from services import NotFoundError, ValidationError


@pytest.fixture
def user_service():
    connection = sqlite3.connect(":memory:")
    repository = UserRepository(connection)
    repository.create_table()
    return UserService(repository)


def valid_user():
    return {
        "username": 'username 1',
        "email": 'paula.carter1@example.com',
        "full_name": 'Rosa King',
        "is_active": False,
        "created_at": '2026-12-07T22:08:00',
    }


def test_create_user_returns_an_id(user_service):
    created = user_service.create_user(valid_user())
    assert created["id"] is not None


def test_get_user_returns_the_created_record(user_service):
    created = user_service.create_user(valid_user())
    fetched = user_service.get_user(created["id"])
    assert fetched["username"] == created["username"]


def test_get_missing_user_raises(user_service):
    with pytest.raises(NotFoundError):
        user_service.get_user(12345)


def test_list_users_pages(user_service):
    for _ in range(25):
        user_service.create_user(valid_user())
    page = user_service.list_users(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_user(user_service):
    created = user_service.create_user(valid_user())
    user_service.delete_user(created["id"])
    with pytest.raises(NotFoundError):
        user_service.get_user(created["id"])


def test_create_user_without_username_is_invalid(user_service):
    data = valid_user()
    data["username"] = ""
    with pytest.raises(ValidationError) as error:
        user_service.create_user(data)
    assert "username is required" in error.value.errors


@pytest.fixture
def product_service():
    connection = sqlite3.connect(":memory:")
    repository = ProductRepository(connection)
    repository.create_table()
    return ProductService(repository)


def valid_product():
    return {
        "sku": 'sku 1',
        "name": 'Oscar Taylor',
        "description": 'description 1',
        "price": 922.9,
        "stock": 218,
    }


def test_create_product_returns_an_id(product_service):
    created = product_service.create_product(valid_product())
    assert created["id"] is not None


def test_get_product_returns_the_created_record(product_service):
    created = product_service.create_product(valid_product())
    fetched = product_service.get_product(created["id"])
    assert fetched["sku"] == created["sku"]


def test_get_missing_product_raises(product_service):
    with pytest.raises(NotFoundError):
        product_service.get_product(12345)


def test_list_products_pages(product_service):
    for _ in range(25):
        product_service.create_product(valid_product())
    page = product_service.list_products(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_product(product_service):
    created = product_service.create_product(valid_product())
    product_service.delete_product(created["id"])
    with pytest.raises(NotFoundError):
        product_service.get_product(created["id"])


def test_create_product_without_sku_is_invalid(product_service):
    data = valid_product()
    data["sku"] = ""
    with pytest.raises(ValidationError) as error:
        product_service.create_product(data)
    assert "sku is required" in error.value.errors


@pytest.fixture
def order_service():
    connection = sqlite3.connect(":memory:")
    repository = OrderRepository(connection)
    repository.create_table()
    return OrderService(repository)


def valid_order():
    return {
        "customer_id": 69,
        "status": 'active',
        "total": 309.21,
        "currency": 'currency 1',
        "placed_at": '2026-08-12T10:10:00',
    }


def test_create_order_returns_an_id(order_service):
    created = order_service.create_order(valid_order())
    assert created["id"] is not None


def test_get_order_returns_the_created_record(order_service):
    created = order_service.create_order(valid_order())
    fetched = order_service.get_order(created["id"])
    assert fetched["customer_id"] == created["customer_id"]


def test_get_missing_order_raises(order_service):
    with pytest.raises(NotFoundError):
        order_service.get_order(12345)


def test_list_orders_pages(order_service):
    for _ in range(25):
        order_service.create_order(valid_order())
    page = order_service.list_orders(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_order(order_service):
    created = order_service.create_order(valid_order())
    order_service.delete_order(created["id"])
    with pytest.raises(NotFoundError):
        order_service.get_order(created["id"])


def test_create_order_without_status_is_invalid(order_service):
    data = valid_order()
    data["status"] = ""
    with pytest.raises(ValidationError) as error:
        order_service.create_order(data)
    assert "status is required" in error.value.errors


@pytest.fixture
def order_item_service():
    connection = sqlite3.connect(":memory:")
    repository = OrderItemRepository(connection)
    repository.create_table()
    return OrderItemService(repository)


def valid_order_item():
    return {
        "order_id": 299,
        "product_id": 337,
        "quantity": 281,
        "unit_price": 299.29,
    }


def test_create_order_item_returns_an_id(order_item_service):
    created = order_item_service.create_order_item(valid_order_item())
    assert created["id"] is not None


def test_get_order_item_returns_the_created_record(order_item_service):
    created = order_item_service.create_order_item(valid_order_item())
    fetched = order_item_service.get_order_item(created["id"])
    assert fetched["order_id"] == created["order_id"]


def test_get_missing_order_item_raises(order_item_service):
    with pytest.raises(NotFoundError):
        order_item_service.get_order_item(12345)


def test_list_order_items_pages(order_item_service):
    for _ in range(25):
        order_item_service.create_order_item(valid_order_item())
    page = order_item_service.list_order_items(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_order_item(order_item_service):
    created = order_item_service.create_order_item(valid_order_item())
    order_item_service.delete_order_item(created["id"])
    with pytest.raises(NotFoundError):
        order_item_service.get_order_item(created["id"])


@pytest.fixture
def customer_service():
    connection = sqlite3.connect(":memory:")
    repository = CustomerRepository(connection)
    repository.create_table()
    return CustomerService(repository)


def valid_customer():
    return {
        "first_name": 'Jack',
        "last_name": 'Johnson',
        "email": 'david.evans1@example.com',
        "phone": '+49 30 6803771',
        "country": 'IE',
    }


def test_create_customer_returns_an_id(customer_service):
    created = customer_service.create_customer(valid_customer())
    assert created["id"] is not None


def test_get_customer_returns_the_created_record(customer_service):
    created = customer_service.create_customer(valid_customer())
    fetched = customer_service.get_customer(created["id"])
    assert fetched["first_name"] == created["first_name"]


def test_get_missing_customer_raises(customer_service):
    with pytest.raises(NotFoundError):
        customer_service.get_customer(12345)


def test_list_customers_pages(customer_service):
    for _ in range(25):
        customer_service.create_customer(valid_customer())
    page = customer_service.list_customers(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_customer(customer_service):
    created = customer_service.create_customer(valid_customer())
    customer_service.delete_customer(created["id"])
    with pytest.raises(NotFoundError):
        customer_service.get_customer(created["id"])


def test_create_customer_without_first_name_is_invalid(customer_service):
    data = valid_customer()
    data["first_name"] = ""
    with pytest.raises(ValidationError) as error:
        customer_service.create_customer(data)
    assert "first_name is required" in error.value.errors


@pytest.fixture
def invoice_service():
    connection = sqlite3.connect(":memory:")
    repository = InvoiceRepository(connection)
    repository.create_table()
    return InvoiceService(repository)


def valid_invoice():
    return {
        "order_id": 465,
        "number": 'number 1',
        "amount": 782.05,
        "due_date": '2026-06-11T12:44:00',
        "paid": False,
    }


def test_create_invoice_returns_an_id(invoice_service):
    created = invoice_service.create_invoice(valid_invoice())
    assert created["id"] is not None


def test_get_invoice_returns_the_created_record(invoice_service):
    created = invoice_service.create_invoice(valid_invoice())
    fetched = invoice_service.get_invoice(created["id"])
    assert fetched["order_id"] == created["order_id"]


def test_get_missing_invoice_raises(invoice_service):
    with pytest.raises(NotFoundError):
        invoice_service.get_invoice(12345)


def test_list_invoices_pages(invoice_service):
    for _ in range(25):
        invoice_service.create_invoice(valid_invoice())
    page = invoice_service.list_invoices(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_invoice(invoice_service):
    created = invoice_service.create_invoice(valid_invoice())
    invoice_service.delete_invoice(created["id"])
    with pytest.raises(NotFoundError):
        invoice_service.get_invoice(created["id"])


def test_create_invoice_without_number_is_invalid(invoice_service):
    data = valid_invoice()
    data["number"] = ""
    with pytest.raises(ValidationError) as error:
        invoice_service.create_invoice(data)
    assert "number is required" in error.value.errors


@pytest.fixture
def payment_service():
    connection = sqlite3.connect(":memory:")
    repository = PaymentRepository(connection)
    repository.create_table()
    return PaymentService(repository)


def valid_payment():
    return {
        "invoice_id": 189,
        "method": 'method 1',
        "amount": 16.48,
        "reference": 'reference 1',
        "received_at": '2026-09-19T07:22:00',
    }


def test_create_payment_returns_an_id(payment_service):
    created = payment_service.create_payment(valid_payment())
    assert created["id"] is not None


def test_get_payment_returns_the_created_record(payment_service):
    created = payment_service.create_payment(valid_payment())
    fetched = payment_service.get_payment(created["id"])
    assert fetched["invoice_id"] == created["invoice_id"]


def test_get_missing_payment_raises(payment_service):
    with pytest.raises(NotFoundError):
        payment_service.get_payment(12345)


def test_list_payments_pages(payment_service):
    for _ in range(25):
        payment_service.create_payment(valid_payment())
    page = payment_service.list_payments(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_payment(payment_service):
    created = payment_service.create_payment(valid_payment())
    payment_service.delete_payment(created["id"])
    with pytest.raises(NotFoundError):
        payment_service.get_payment(created["id"])


def test_create_payment_without_method_is_invalid(payment_service):
    data = valid_payment()
    data["method"] = ""
    with pytest.raises(ValidationError) as error:
        payment_service.create_payment(data)
    assert "method is required" in error.value.errors


@pytest.fixture
def category_service():
    connection = sqlite3.connect(":memory:")
    repository = CategoryRepository(connection)
    repository.create_table()
    return CategoryService(repository)


def valid_category():
    return {
        "name": 'Henry Wright',
        "slug": 'slug 1',
        "parent_id": 254,
        "position": 28,
    }


def test_create_category_returns_an_id(category_service):
    created = category_service.create_category(valid_category())
    assert created["id"] is not None


def test_get_category_returns_the_created_record(category_service):
    created = category_service.create_category(valid_category())
    fetched = category_service.get_category(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_category_raises(category_service):
    with pytest.raises(NotFoundError):
        category_service.get_category(12345)


def test_list_categories_pages(category_service):
    for _ in range(25):
        category_service.create_category(valid_category())
    page = category_service.list_categories(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_category(category_service):
    created = category_service.create_category(valid_category())
    category_service.delete_category(created["id"])
    with pytest.raises(NotFoundError):
        category_service.get_category(created["id"])


def test_create_category_without_name_is_invalid(category_service):
    data = valid_category()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        category_service.create_category(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def review_service():
    connection = sqlite3.connect(":memory:")
    repository = ReviewRepository(connection)
    repository.create_table()
    return ReviewService(repository)


def valid_review():
    return {
        "product_id": 374,
        "user_id": 216,
        "rating": 189,
        "title": 'title 1',
        "body": 'body 1',
    }


def test_create_review_returns_an_id(review_service):
    created = review_service.create_review(valid_review())
    assert created["id"] is not None


def test_get_review_returns_the_created_record(review_service):
    created = review_service.create_review(valid_review())
    fetched = review_service.get_review(created["id"])
    assert fetched["product_id"] == created["product_id"]


def test_get_missing_review_raises(review_service):
    with pytest.raises(NotFoundError):
        review_service.get_review(12345)


def test_list_reviews_pages(review_service):
    for _ in range(25):
        review_service.create_review(valid_review())
    page = review_service.list_reviews(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_review(review_service):
    created = review_service.create_review(valid_review())
    review_service.delete_review(created["id"])
    with pytest.raises(NotFoundError):
        review_service.get_review(created["id"])


def test_create_review_without_title_is_invalid(review_service):
    data = valid_review()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        review_service.create_review(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def address_service():
    connection = sqlite3.connect(":memory:")
    repository = AddressRepository(connection)
    repository.create_table()
    return AddressService(repository)


def valid_address():
    return {
        "customer_id": 23,
        "street": 'street 1',
        "city": 'Munich',
        "postal_code": 'postal code 1',
        "country": 'PT',
    }


def test_create_address_returns_an_id(address_service):
    created = address_service.create_address(valid_address())
    assert created["id"] is not None


def test_get_address_returns_the_created_record(address_service):
    created = address_service.create_address(valid_address())
    fetched = address_service.get_address(created["id"])
    assert fetched["customer_id"] == created["customer_id"]


def test_get_missing_address_raises(address_service):
    with pytest.raises(NotFoundError):
        address_service.get_address(12345)


def test_list_addresses_pages(address_service):
    for _ in range(25):
        address_service.create_address(valid_address())
    page = address_service.list_addresses(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_address(address_service):
    created = address_service.create_address(valid_address())
    address_service.delete_address(created["id"])
    with pytest.raises(NotFoundError):
        address_service.get_address(created["id"])


def test_create_address_without_street_is_invalid(address_service):
    data = valid_address()
    data["street"] = ""
    with pytest.raises(ValidationError) as error:
        address_service.create_address(data)
    assert "street is required" in error.value.errors


@pytest.fixture
def shipment_service():
    connection = sqlite3.connect(":memory:")
    repository = ShipmentRepository(connection)
    repository.create_table()
    return ShipmentService(repository)


def valid_shipment():
    return {
        "order_id": 269,
        "carrier": 'carrier 1',
        "tracking_number": 'tracking number 1',
        "shipped_at": '2026-06-27T15:29:00',
        "delivered": False,
    }


def test_create_shipment_returns_an_id(shipment_service):
    created = shipment_service.create_shipment(valid_shipment())
    assert created["id"] is not None


def test_get_shipment_returns_the_created_record(shipment_service):
    created = shipment_service.create_shipment(valid_shipment())
    fetched = shipment_service.get_shipment(created["id"])
    assert fetched["order_id"] == created["order_id"]


def test_get_missing_shipment_raises(shipment_service):
    with pytest.raises(NotFoundError):
        shipment_service.get_shipment(12345)


def test_list_shipments_pages(shipment_service):
    for _ in range(25):
        shipment_service.create_shipment(valid_shipment())
    page = shipment_service.list_shipments(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_shipment(shipment_service):
    created = shipment_service.create_shipment(valid_shipment())
    shipment_service.delete_shipment(created["id"])
    with pytest.raises(NotFoundError):
        shipment_service.get_shipment(created["id"])


def test_create_shipment_without_carrier_is_invalid(shipment_service):
    data = valid_shipment()
    data["carrier"] = ""
    with pytest.raises(ValidationError) as error:
        shipment_service.create_shipment(data)
    assert "carrier is required" in error.value.errors


@pytest.fixture
def warehouse_service():
    connection = sqlite3.connect(":memory:")
    repository = WarehouseRepository(connection)
    repository.create_table()
    return WarehouseService(repository)


def valid_warehouse():
    return {
        "name": 'Nora Lewis',
        "city": 'Zurich',
        "capacity": 328,
        "is_open": False,
    }


def test_create_warehouse_returns_an_id(warehouse_service):
    created = warehouse_service.create_warehouse(valid_warehouse())
    assert created["id"] is not None


def test_get_warehouse_returns_the_created_record(warehouse_service):
    created = warehouse_service.create_warehouse(valid_warehouse())
    fetched = warehouse_service.get_warehouse(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_warehouse_raises(warehouse_service):
    with pytest.raises(NotFoundError):
        warehouse_service.get_warehouse(12345)


def test_list_warehouses_pages(warehouse_service):
    for _ in range(25):
        warehouse_service.create_warehouse(valid_warehouse())
    page = warehouse_service.list_warehouses(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_warehouse(warehouse_service):
    created = warehouse_service.create_warehouse(valid_warehouse())
    warehouse_service.delete_warehouse(created["id"])
    with pytest.raises(NotFoundError):
        warehouse_service.get_warehouse(created["id"])


def test_create_warehouse_without_name_is_invalid(warehouse_service):
    data = valid_warehouse()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        warehouse_service.create_warehouse(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def supplier_service():
    connection = sqlite3.connect(":memory:")
    repository = SupplierRepository(connection)
    repository.create_table()
    return SupplierService(repository)


def valid_supplier():
    return {
        "name": 'Anna Collins',
        "contact_email": 'tara.collins1@example.com',
        "phone": '+49 30 8705272',
        "country": 'DE',
        "rating": 265,
    }


def test_create_supplier_returns_an_id(supplier_service):
    created = supplier_service.create_supplier(valid_supplier())
    assert created["id"] is not None


def test_get_supplier_returns_the_created_record(supplier_service):
    created = supplier_service.create_supplier(valid_supplier())
    fetched = supplier_service.get_supplier(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_supplier_raises(supplier_service):
    with pytest.raises(NotFoundError):
        supplier_service.get_supplier(12345)


def test_list_suppliers_pages(supplier_service):
    for _ in range(25):
        supplier_service.create_supplier(valid_supplier())
    page = supplier_service.list_suppliers(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_supplier(supplier_service):
    created = supplier_service.create_supplier(valid_supplier())
    supplier_service.delete_supplier(created["id"])
    with pytest.raises(NotFoundError):
        supplier_service.get_supplier(created["id"])


def test_create_supplier_without_name_is_invalid(supplier_service):
    data = valid_supplier()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        supplier_service.create_supplier(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def coupon_service():
    connection = sqlite3.connect(":memory:")
    repository = CouponRepository(connection)
    repository.create_table()
    return CouponService(repository)


def valid_coupon():
    return {
        "code": 'code 1',
        "discount_percent": 158,
        "valid_until": '2026-11-04T06:37:00',
        "max_uses": 88,
        "is_active": True,
    }


def test_create_coupon_returns_an_id(coupon_service):
    created = coupon_service.create_coupon(valid_coupon())
    assert created["id"] is not None


def test_get_coupon_returns_the_created_record(coupon_service):
    created = coupon_service.create_coupon(valid_coupon())
    fetched = coupon_service.get_coupon(created["id"])
    assert fetched["code"] == created["code"]


def test_get_missing_coupon_raises(coupon_service):
    with pytest.raises(NotFoundError):
        coupon_service.get_coupon(12345)


def test_list_coupons_pages(coupon_service):
    for _ in range(25):
        coupon_service.create_coupon(valid_coupon())
    page = coupon_service.list_coupons(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_coupon(coupon_service):
    created = coupon_service.create_coupon(valid_coupon())
    coupon_service.delete_coupon(created["id"])
    with pytest.raises(NotFoundError):
        coupon_service.get_coupon(created["id"])


def test_create_coupon_without_code_is_invalid(coupon_service):
    data = valid_coupon()
    data["code"] = ""
    with pytest.raises(ValidationError) as error:
        coupon_service.create_coupon(data)
    assert "code is required" in error.value.errors


@pytest.fixture
def employee_service():
    connection = sqlite3.connect(":memory:")
    repository = EmployeeRepository(connection)
    repository.create_table()
    return EmployeeService(repository)


def valid_employee():
    return {
        "first_name": 'Emma',
        "last_name": 'Hall',
        "email": 'jack.collins1@example.com',
        "department": 'department 1',
        "salary": 337.09,
    }


def test_create_employee_returns_an_id(employee_service):
    created = employee_service.create_employee(valid_employee())
    assert created["id"] is not None


def test_get_employee_returns_the_created_record(employee_service):
    created = employee_service.create_employee(valid_employee())
    fetched = employee_service.get_employee(created["id"])
    assert fetched["first_name"] == created["first_name"]


def test_get_missing_employee_raises(employee_service):
    with pytest.raises(NotFoundError):
        employee_service.get_employee(12345)


def test_list_employees_pages(employee_service):
    for _ in range(25):
        employee_service.create_employee(valid_employee())
    page = employee_service.list_employees(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_employee(employee_service):
    created = employee_service.create_employee(valid_employee())
    employee_service.delete_employee(created["id"])
    with pytest.raises(NotFoundError):
        employee_service.get_employee(created["id"])


def test_create_employee_without_first_name_is_invalid(employee_service):
    data = valid_employee()
    data["first_name"] = ""
    with pytest.raises(ValidationError) as error:
        employee_service.create_employee(data)
    assert "first_name is required" in error.value.errors


@pytest.fixture
def department_service():
    connection = sqlite3.connect(":memory:")
    repository = DepartmentRepository(connection)
    repository.create_table()
    return DepartmentService(repository)


def valid_department():
    return {
        "name": 'Emma Smith',
        "code": 'code 1',
        "manager_id": 24,
        "budget": 991.93,
    }


def test_create_department_returns_an_id(department_service):
    created = department_service.create_department(valid_department())
    assert created["id"] is not None


def test_get_department_returns_the_created_record(department_service):
    created = department_service.create_department(valid_department())
    fetched = department_service.get_department(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_department_raises(department_service):
    with pytest.raises(NotFoundError):
        department_service.get_department(12345)


def test_list_departments_pages(department_service):
    for _ in range(25):
        department_service.create_department(valid_department())
    page = department_service.list_departments(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_department(department_service):
    created = department_service.create_department(valid_department())
    department_service.delete_department(created["id"])
    with pytest.raises(NotFoundError):
        department_service.get_department(created["id"])


def test_create_department_without_name_is_invalid(department_service):
    data = valid_department()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        department_service.create_department(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def project_service():
    connection = sqlite3.connect(":memory:")
    repository = ProjectRepository(connection)
    repository.create_table()
    return ProjectService(repository)


def valid_project():
    return {
        "name": 'Quinn Lewis',
        "description": 'description 1',
        "start_date": '2026-01-09T22:04:00',
        "end_date": '2026-03-24T00:59:00',
        "budget": 79.04,
    }


def test_create_project_returns_an_id(project_service):
    created = project_service.create_project(valid_project())
    assert created["id"] is not None


def test_get_project_returns_the_created_record(project_service):
    created = project_service.create_project(valid_project())
    fetched = project_service.get_project(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_project_raises(project_service):
    with pytest.raises(NotFoundError):
        project_service.get_project(12345)


def test_list_projects_pages(project_service):
    for _ in range(25):
        project_service.create_project(valid_project())
    page = project_service.list_projects(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_project(project_service):
    created = project_service.create_project(valid_project())
    project_service.delete_project(created["id"])
    with pytest.raises(NotFoundError):
        project_service.get_project(created["id"])


def test_create_project_without_name_is_invalid(project_service):
    data = valid_project()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        project_service.create_project(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def task_service():
    connection = sqlite3.connect(":memory:")
    repository = TaskRepository(connection)
    repository.create_table()
    return TaskService(repository)


def valid_task():
    return {
        "project_id": 467,
        "title": 'title 1',
        "status": 'done',
        "priority": 193,
        "due_date": '2026-04-24T19:19:00',
    }


def test_create_task_returns_an_id(task_service):
    created = task_service.create_task(valid_task())
    assert created["id"] is not None


def test_get_task_returns_the_created_record(task_service):
    created = task_service.create_task(valid_task())
    fetched = task_service.get_task(created["id"])
    assert fetched["project_id"] == created["project_id"]


def test_get_missing_task_raises(task_service):
    with pytest.raises(NotFoundError):
        task_service.get_task(12345)


def test_list_tasks_pages(task_service):
    for _ in range(25):
        task_service.create_task(valid_task())
    page = task_service.list_tasks(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_task(task_service):
    created = task_service.create_task(valid_task())
    task_service.delete_task(created["id"])
    with pytest.raises(NotFoundError):
        task_service.get_task(created["id"])


def test_create_task_without_title_is_invalid(task_service):
    data = valid_task()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        task_service.create_task(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def comment_service():
    connection = sqlite3.connect(":memory:")
    repository = CommentRepository(connection)
    repository.create_table()
    return CommentService(repository)


def valid_comment():
    return {
        "task_id": 28,
        "author_id": 183,
        "body": 'body 1',
        "created_at": '2026-09-02T04:58:00',
    }


def test_create_comment_returns_an_id(comment_service):
    created = comment_service.create_comment(valid_comment())
    assert created["id"] is not None


def test_get_comment_returns_the_created_record(comment_service):
    created = comment_service.create_comment(valid_comment())
    fetched = comment_service.get_comment(created["id"])
    assert fetched["task_id"] == created["task_id"]


def test_get_missing_comment_raises(comment_service):
    with pytest.raises(NotFoundError):
        comment_service.get_comment(12345)


def test_list_comments_pages(comment_service):
    for _ in range(25):
        comment_service.create_comment(valid_comment())
    page = comment_service.list_comments(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_comment(comment_service):
    created = comment_service.create_comment(valid_comment())
    comment_service.delete_comment(created["id"])
    with pytest.raises(NotFoundError):
        comment_service.get_comment(created["id"])


def test_create_comment_without_body_is_invalid(comment_service):
    data = valid_comment()
    data["body"] = ""
    with pytest.raises(ValidationError) as error:
        comment_service.create_comment(data)
    assert "body is required" in error.value.errors


@pytest.fixture
def tag_service():
    connection = sqlite3.connect(":memory:")
    repository = TagRepository(connection)
    repository.create_table()
    return TagService(repository)


def valid_tag():
    return {
        "name": 'David Johnson',
        "color": 'color 1',
        "usage_count": 187,
    }


def test_create_tag_returns_an_id(tag_service):
    created = tag_service.create_tag(valid_tag())
    assert created["id"] is not None


def test_get_tag_returns_the_created_record(tag_service):
    created = tag_service.create_tag(valid_tag())
    fetched = tag_service.get_tag(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_tag_raises(tag_service):
    with pytest.raises(NotFoundError):
        tag_service.get_tag(12345)


def test_list_tags_pages(tag_service):
    for _ in range(25):
        tag_service.create_tag(valid_tag())
    page = tag_service.list_tags(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_tag(tag_service):
    created = tag_service.create_tag(valid_tag())
    tag_service.delete_tag(created["id"])
    with pytest.raises(NotFoundError):
        tag_service.get_tag(created["id"])


def test_create_tag_without_name_is_invalid(tag_service):
    data = valid_tag()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        tag_service.create_tag(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def article_service():
    connection = sqlite3.connect(":memory:")
    repository = ArticleRepository(connection)
    repository.create_table()
    return ArticleService(repository)


def valid_article():
    return {
        "title": 'title 1',
        "slug": 'slug 1',
        "body": 'body 1',
        "author_id": 5,
        "published": True,
    }


def test_create_article_returns_an_id(article_service):
    created = article_service.create_article(valid_article())
    assert created["id"] is not None


def test_get_article_returns_the_created_record(article_service):
    created = article_service.create_article(valid_article())
    fetched = article_service.get_article(created["id"])
    assert fetched["title"] == created["title"]


def test_get_missing_article_raises(article_service):
    with pytest.raises(NotFoundError):
        article_service.get_article(12345)


def test_list_articles_pages(article_service):
    for _ in range(25):
        article_service.create_article(valid_article())
    page = article_service.list_articles(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_article(article_service):
    created = article_service.create_article(valid_article())
    article_service.delete_article(created["id"])
    with pytest.raises(NotFoundError):
        article_service.get_article(created["id"])


def test_create_article_without_title_is_invalid(article_service):
    data = valid_article()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        article_service.create_article(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def author_service():
    connection = sqlite3.connect(":memory:")
    repository = AuthorRepository(connection)
    repository.create_table()
    return AuthorService(repository)


def valid_author():
    return {
        "name": 'Felix Nelson',
        "bio": 'bio 1',
        "email": 'anna.baker1@example.com',
        "website": 'website 1',
    }


def test_create_author_returns_an_id(author_service):
    created = author_service.create_author(valid_author())
    assert created["id"] is not None


def test_get_author_returns_the_created_record(author_service):
    created = author_service.create_author(valid_author())
    fetched = author_service.get_author(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_author_raises(author_service):
    with pytest.raises(NotFoundError):
        author_service.get_author(12345)


def test_list_authors_pages(author_service):
    for _ in range(25):
        author_service.create_author(valid_author())
    page = author_service.list_authors(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_author(author_service):
    created = author_service.create_author(valid_author())
    author_service.delete_author(created["id"])
    with pytest.raises(NotFoundError):
        author_service.get_author(created["id"])


def test_create_author_without_name_is_invalid(author_service):
    data = valid_author()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        author_service.create_author(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def book_service():
    connection = sqlite3.connect(":memory:")
    repository = BookRepository(connection)
    repository.create_table()
    return BookService(repository)


def valid_book():
    return {
        "title": 'title 1',
        "isbn": 'isbn 1',
        "author_id": 277,
        "year": 388,
        "pages": 448,
    }


def test_create_book_returns_an_id(book_service):
    created = book_service.create_book(valid_book())
    assert created["id"] is not None


def test_get_book_returns_the_created_record(book_service):
    created = book_service.create_book(valid_book())
    fetched = book_service.get_book(created["id"])
    assert fetched["title"] == created["title"]


def test_get_missing_book_raises(book_service):
    with pytest.raises(NotFoundError):
        book_service.get_book(12345)


def test_list_books_pages(book_service):
    for _ in range(25):
        book_service.create_book(valid_book())
    page = book_service.list_books(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_book(book_service):
    created = book_service.create_book(valid_book())
    book_service.delete_book(created["id"])
    with pytest.raises(NotFoundError):
        book_service.get_book(created["id"])


def test_create_book_without_title_is_invalid(book_service):
    data = valid_book()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        book_service.create_book(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def library_service():
    connection = sqlite3.connect(":memory:")
    repository = LibraryRepository(connection)
    repository.create_table()
    return LibraryService(repository)


def valid_library():
    return {
        "name": 'Nora Parker',
        "city": 'Amsterdam',
        "opening_hour": 476,
        "closing_hour": 274,
    }


def test_create_library_returns_an_id(library_service):
    created = library_service.create_library(valid_library())
    assert created["id"] is not None


def test_get_library_returns_the_created_record(library_service):
    created = library_service.create_library(valid_library())
    fetched = library_service.get_library(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_library_raises(library_service):
    with pytest.raises(NotFoundError):
        library_service.get_library(12345)


def test_list_libraries_pages(library_service):
    for _ in range(25):
        library_service.create_library(valid_library())
    page = library_service.list_libraries(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_library(library_service):
    created = library_service.create_library(valid_library())
    library_service.delete_library(created["id"])
    with pytest.raises(NotFoundError):
        library_service.get_library(created["id"])


def test_create_library_without_name_is_invalid(library_service):
    data = valid_library()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        library_service.create_library(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def loan_service():
    connection = sqlite3.connect(":memory:")
    repository = LoanRepository(connection)
    repository.create_table()
    return LoanService(repository)


def valid_loan():
    return {
        "book_id": 478,
        "member_id": 436,
        "borrowed_at": '2026-09-14T17:46:00',
        "due_date": '2026-06-11T18:29:00',
        "returned": True,
    }


def test_create_loan_returns_an_id(loan_service):
    created = loan_service.create_loan(valid_loan())
    assert created["id"] is not None


def test_get_loan_returns_the_created_record(loan_service):
    created = loan_service.create_loan(valid_loan())
    fetched = loan_service.get_loan(created["id"])
    assert fetched["book_id"] == created["book_id"]


def test_get_missing_loan_raises(loan_service):
    with pytest.raises(NotFoundError):
        loan_service.get_loan(12345)


def test_list_loans_pages(loan_service):
    for _ in range(25):
        loan_service.create_loan(valid_loan())
    page = loan_service.list_loans(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_loan(loan_service):
    created = loan_service.create_loan(valid_loan())
    loan_service.delete_loan(created["id"])
    with pytest.raises(NotFoundError):
        loan_service.get_loan(created["id"])


@pytest.fixture
def member_service():
    connection = sqlite3.connect(":memory:")
    repository = MemberRepository(connection)
    repository.create_table()
    return MemberService(repository)


def valid_member():
    return {
        "first_name": 'David',
        "last_name": 'Nelson',
        "email": 'david.brown1@example.com',
        "joined_at": '2026-06-13T14:57:00',
    }


def test_create_member_returns_an_id(member_service):
    created = member_service.create_member(valid_member())
    assert created["id"] is not None


def test_get_member_returns_the_created_record(member_service):
    created = member_service.create_member(valid_member())
    fetched = member_service.get_member(created["id"])
    assert fetched["first_name"] == created["first_name"]


def test_get_missing_member_raises(member_service):
    with pytest.raises(NotFoundError):
        member_service.get_member(12345)


def test_list_members_pages(member_service):
    for _ in range(25):
        member_service.create_member(valid_member())
    page = member_service.list_members(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_member(member_service):
    created = member_service.create_member(valid_member())
    member_service.delete_member(created["id"])
    with pytest.raises(NotFoundError):
        member_service.get_member(created["id"])


def test_create_member_without_first_name_is_invalid(member_service):
    data = valid_member()
    data["first_name"] = ""
    with pytest.raises(ValidationError) as error:
        member_service.create_member(data)
    assert "first_name is required" in error.value.errors


@pytest.fixture
def event_service():
    connection = sqlite3.connect(":memory:")
    repository = EventRepository(connection)
    repository.create_table()
    return EventService(repository)


def valid_event():
    return {
        "title": 'title 1',
        "location": 'location 1',
        "starts_at": '2026-04-22T10:10:00',
        "ends_at": '2026-01-27T19:56:00',
        "capacity": 394,
    }


def test_create_event_returns_an_id(event_service):
    created = event_service.create_event(valid_event())
    assert created["id"] is not None


def test_get_event_returns_the_created_record(event_service):
    created = event_service.create_event(valid_event())
    fetched = event_service.get_event(created["id"])
    assert fetched["title"] == created["title"]


def test_get_missing_event_raises(event_service):
    with pytest.raises(NotFoundError):
        event_service.get_event(12345)


def test_list_events_pages(event_service):
    for _ in range(25):
        event_service.create_event(valid_event())
    page = event_service.list_events(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_event(event_service):
    created = event_service.create_event(valid_event())
    event_service.delete_event(created["id"])
    with pytest.raises(NotFoundError):
        event_service.get_event(created["id"])


def test_create_event_without_title_is_invalid(event_service):
    data = valid_event()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        event_service.create_event(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def ticket_service():
    connection = sqlite3.connect(":memory:")
    repository = TicketRepository(connection)
    repository.create_table()
    return TicketService(repository)


def valid_ticket():
    return {
        "event_id": 383,
        "holder_name": 'Lena Baker',
        "seat": 'seat 1',
        "price": 641.75,
        "checked_in": False,
    }


def test_create_ticket_returns_an_id(ticket_service):
    created = ticket_service.create_ticket(valid_ticket())
    assert created["id"] is not None


def test_get_ticket_returns_the_created_record(ticket_service):
    created = ticket_service.create_ticket(valid_ticket())
    fetched = ticket_service.get_ticket(created["id"])
    assert fetched["event_id"] == created["event_id"]


def test_get_missing_ticket_raises(ticket_service):
    with pytest.raises(NotFoundError):
        ticket_service.get_ticket(12345)


def test_list_tickets_pages(ticket_service):
    for _ in range(25):
        ticket_service.create_ticket(valid_ticket())
    page = ticket_service.list_tickets(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_ticket(ticket_service):
    created = ticket_service.create_ticket(valid_ticket())
    ticket_service.delete_ticket(created["id"])
    with pytest.raises(NotFoundError):
        ticket_service.get_ticket(created["id"])


def test_create_ticket_without_holder_name_is_invalid(ticket_service):
    data = valid_ticket()
    data["holder_name"] = ""
    with pytest.raises(ValidationError) as error:
        ticket_service.create_ticket(data)
    assert "holder_name is required" in error.value.errors


@pytest.fixture
def venue_service():
    connection = sqlite3.connect(":memory:")
    repository = VenueRepository(connection)
    repository.create_table()
    return VenueService(repository)


def valid_venue():
    return {
        "name": 'Noah Turner',
        "address": 'address 1',
        "city": 'Stockholm',
        "capacity": 179,
    }


def test_create_venue_returns_an_id(venue_service):
    created = venue_service.create_venue(valid_venue())
    assert created["id"] is not None


def test_get_venue_returns_the_created_record(venue_service):
    created = venue_service.create_venue(valid_venue())
    fetched = venue_service.get_venue(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_venue_raises(venue_service):
    with pytest.raises(NotFoundError):
        venue_service.get_venue(12345)


def test_list_venues_pages(venue_service):
    for _ in range(25):
        venue_service.create_venue(valid_venue())
    page = venue_service.list_venues(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_venue(venue_service):
    created = venue_service.create_venue(valid_venue())
    venue_service.delete_venue(created["id"])
    with pytest.raises(NotFoundError):
        venue_service.get_venue(created["id"])


def test_create_venue_without_name_is_invalid(venue_service):
    data = valid_venue()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        venue_service.create_venue(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def vehicle_service():
    connection = sqlite3.connect(":memory:")
    repository = VehicleRepository(connection)
    repository.create_table()
    return VehicleService(repository)


def valid_vehicle():
    return {
        "plate": 'plate 1',
        "make": 'make 1',
        "model": 'model 1',
        "year": 144,
        "mileage": 238,
    }


def test_create_vehicle_returns_an_id(vehicle_service):
    created = vehicle_service.create_vehicle(valid_vehicle())
    assert created["id"] is not None


def test_get_vehicle_returns_the_created_record(vehicle_service):
    created = vehicle_service.create_vehicle(valid_vehicle())
    fetched = vehicle_service.get_vehicle(created["id"])
    assert fetched["plate"] == created["plate"]


def test_get_missing_vehicle_raises(vehicle_service):
    with pytest.raises(NotFoundError):
        vehicle_service.get_vehicle(12345)


def test_list_vehicles_pages(vehicle_service):
    for _ in range(25):
        vehicle_service.create_vehicle(valid_vehicle())
    page = vehicle_service.list_vehicles(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_vehicle(vehicle_service):
    created = vehicle_service.create_vehicle(valid_vehicle())
    vehicle_service.delete_vehicle(created["id"])
    with pytest.raises(NotFoundError):
        vehicle_service.get_vehicle(created["id"])


def test_create_vehicle_without_plate_is_invalid(vehicle_service):
    data = valid_vehicle()
    data["plate"] = ""
    with pytest.raises(ValidationError) as error:
        vehicle_service.create_vehicle(data)
    assert "plate is required" in error.value.errors


@pytest.fixture
def driver_service():
    connection = sqlite3.connect(":memory:")
    repository = DriverRepository(connection)
    repository.create_table()
    return DriverService(repository)


def valid_driver():
    return {
        "first_name": 'Ben',
        "last_name": 'Nelson',
        "license_number": 'license number 1',
        "rating": 637.91,
    }


def test_create_driver_returns_an_id(driver_service):
    created = driver_service.create_driver(valid_driver())
    assert created["id"] is not None


def test_get_driver_returns_the_created_record(driver_service):
    created = driver_service.create_driver(valid_driver())
    fetched = driver_service.get_driver(created["id"])
    assert fetched["first_name"] == created["first_name"]


def test_get_missing_driver_raises(driver_service):
    with pytest.raises(NotFoundError):
        driver_service.get_driver(12345)


def test_list_drivers_pages(driver_service):
    for _ in range(25):
        driver_service.create_driver(valid_driver())
    page = driver_service.list_drivers(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_driver(driver_service):
    created = driver_service.create_driver(valid_driver())
    driver_service.delete_driver(created["id"])
    with pytest.raises(NotFoundError):
        driver_service.get_driver(created["id"])


def test_create_driver_without_first_name_is_invalid(driver_service):
    data = valid_driver()
    data["first_name"] = ""
    with pytest.raises(ValidationError) as error:
        driver_service.create_driver(data)
    assert "first_name is required" in error.value.errors


@pytest.fixture
def trip_service():
    connection = sqlite3.connect(":memory:")
    repository = TripRepository(connection)
    repository.create_table()
    return TripService(repository)


def valid_trip():
    return {
        "vehicle_id": 221,
        "driver_id": 154,
        "distance_km": 791.54,
        "started_at": '2026-11-27T13:11:00',
        "fare": 238.38,
    }


def test_create_trip_returns_an_id(trip_service):
    created = trip_service.create_trip(valid_trip())
    assert created["id"] is not None


def test_get_trip_returns_the_created_record(trip_service):
    created = trip_service.create_trip(valid_trip())
    fetched = trip_service.get_trip(created["id"])
    assert fetched["vehicle_id"] == created["vehicle_id"]


def test_get_missing_trip_raises(trip_service):
    with pytest.raises(NotFoundError):
        trip_service.get_trip(12345)


def test_list_trips_pages(trip_service):
    for _ in range(25):
        trip_service.create_trip(valid_trip())
    page = trip_service.list_trips(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_trip(trip_service):
    created = trip_service.create_trip(valid_trip())
    trip_service.delete_trip(created["id"])
    with pytest.raises(NotFoundError):
        trip_service.get_trip(created["id"])


@pytest.fixture
def patient_service():
    connection = sqlite3.connect(":memory:")
    repository = PatientRepository(connection)
    repository.create_table()
    return PatientService(repository)


def valid_patient():
    return {
        "first_name": 'Zoe',
        "last_name": 'Johnson',
        "birth_date": '2026-09-12T20:10:00',
        "phone": '+49 30 1303271',
    }


def test_create_patient_returns_an_id(patient_service):
    created = patient_service.create_patient(valid_patient())
    assert created["id"] is not None


def test_get_patient_returns_the_created_record(patient_service):
    created = patient_service.create_patient(valid_patient())
    fetched = patient_service.get_patient(created["id"])
    assert fetched["first_name"] == created["first_name"]


def test_get_missing_patient_raises(patient_service):
    with pytest.raises(NotFoundError):
        patient_service.get_patient(12345)


def test_list_patients_pages(patient_service):
    for _ in range(25):
        patient_service.create_patient(valid_patient())
    page = patient_service.list_patients(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_patient(patient_service):
    created = patient_service.create_patient(valid_patient())
    patient_service.delete_patient(created["id"])
    with pytest.raises(NotFoundError):
        patient_service.get_patient(created["id"])


def test_create_patient_without_first_name_is_invalid(patient_service):
    data = valid_patient()
    data["first_name"] = ""
    with pytest.raises(ValidationError) as error:
        patient_service.create_patient(data)
    assert "first_name is required" in error.value.errors


@pytest.fixture
def doctor_service():
    connection = sqlite3.connect(":memory:")
    repository = DoctorRepository(connection)
    repository.create_table()
    return DoctorService(repository)


def valid_doctor():
    return {
        "first_name": 'Ben',
        "last_name": 'Adams',
        "specialty": 'specialty 1',
        "room": 'room 1',
    }


def test_create_doctor_returns_an_id(doctor_service):
    created = doctor_service.create_doctor(valid_doctor())
    assert created["id"] is not None


def test_get_doctor_returns_the_created_record(doctor_service):
    created = doctor_service.create_doctor(valid_doctor())
    fetched = doctor_service.get_doctor(created["id"])
    assert fetched["first_name"] == created["first_name"]


def test_get_missing_doctor_raises(doctor_service):
    with pytest.raises(NotFoundError):
        doctor_service.get_doctor(12345)


def test_list_doctors_pages(doctor_service):
    for _ in range(25):
        doctor_service.create_doctor(valid_doctor())
    page = doctor_service.list_doctors(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_doctor(doctor_service):
    created = doctor_service.create_doctor(valid_doctor())
    doctor_service.delete_doctor(created["id"])
    with pytest.raises(NotFoundError):
        doctor_service.get_doctor(created["id"])


def test_create_doctor_without_first_name_is_invalid(doctor_service):
    data = valid_doctor()
    data["first_name"] = ""
    with pytest.raises(ValidationError) as error:
        doctor_service.create_doctor(data)
    assert "first_name is required" in error.value.errors


@pytest.fixture
def appointment_service():
    connection = sqlite3.connect(":memory:")
    repository = AppointmentRepository(connection)
    repository.create_table()
    return AppointmentService(repository)


def valid_appointment():
    return {
        "patient_id": 217,
        "doctor_id": 111,
        "scheduled_at": '2026-04-17T08:11:00',
        "reason": 'reason 1',
        "confirmed": False,
    }


def test_create_appointment_returns_an_id(appointment_service):
    created = appointment_service.create_appointment(valid_appointment())
    assert created["id"] is not None


def test_get_appointment_returns_the_created_record(appointment_service):
    created = appointment_service.create_appointment(valid_appointment())
    fetched = appointment_service.get_appointment(created["id"])
    assert fetched["patient_id"] == created["patient_id"]


def test_get_missing_appointment_raises(appointment_service):
    with pytest.raises(NotFoundError):
        appointment_service.get_appointment(12345)


def test_list_appointments_pages(appointment_service):
    for _ in range(25):
        appointment_service.create_appointment(valid_appointment())
    page = appointment_service.list_appointments(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_appointment(appointment_service):
    created = appointment_service.create_appointment(valid_appointment())
    appointment_service.delete_appointment(created["id"])
    with pytest.raises(NotFoundError):
        appointment_service.get_appointment(created["id"])


def test_create_appointment_without_reason_is_invalid(appointment_service):
    data = valid_appointment()
    data["reason"] = ""
    with pytest.raises(ValidationError) as error:
        appointment_service.create_appointment(data)
    assert "reason is required" in error.value.errors


@pytest.fixture
def course_service():
    connection = sqlite3.connect(":memory:")
    repository = CourseRepository(connection)
    repository.create_table()
    return CourseService(repository)


def valid_course():
    return {
        "title": 'title 1',
        "code": 'code 1',
        "credits": 14,
        "teacher_id": 361,
    }


def test_create_course_returns_an_id(course_service):
    created = course_service.create_course(valid_course())
    assert created["id"] is not None


def test_get_course_returns_the_created_record(course_service):
    created = course_service.create_course(valid_course())
    fetched = course_service.get_course(created["id"])
    assert fetched["title"] == created["title"]


def test_get_missing_course_raises(course_service):
    with pytest.raises(NotFoundError):
        course_service.get_course(12345)


def test_list_courses_pages(course_service):
    for _ in range(25):
        course_service.create_course(valid_course())
    page = course_service.list_courses(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_course(course_service):
    created = course_service.create_course(valid_course())
    course_service.delete_course(created["id"])
    with pytest.raises(NotFoundError):
        course_service.get_course(created["id"])


def test_create_course_without_title_is_invalid(course_service):
    data = valid_course()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        course_service.create_course(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def student_service():
    connection = sqlite3.connect(":memory:")
    repository = StudentRepository(connection)
    repository.create_table()
    return StudentService(repository)


def valid_student():
    return {
        "first_name": 'Olivia',
        "last_name": 'Baker',
        "email": 'liam.wilson1@example.com',
        "enrolled_at": '2026-11-26T07:26:00',
    }


def test_create_student_returns_an_id(student_service):
    created = student_service.create_student(valid_student())
    assert created["id"] is not None


def test_get_student_returns_the_created_record(student_service):
    created = student_service.create_student(valid_student())
    fetched = student_service.get_student(created["id"])
    assert fetched["first_name"] == created["first_name"]


def test_get_missing_student_raises(student_service):
    with pytest.raises(NotFoundError):
        student_service.get_student(12345)


def test_list_students_pages(student_service):
    for _ in range(25):
        student_service.create_student(valid_student())
    page = student_service.list_students(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_student(student_service):
    created = student_service.create_student(valid_student())
    student_service.delete_student(created["id"])
    with pytest.raises(NotFoundError):
        student_service.get_student(created["id"])


def test_create_student_without_first_name_is_invalid(student_service):
    data = valid_student()
    data["first_name"] = ""
    with pytest.raises(ValidationError) as error:
        student_service.create_student(data)
    assert "first_name is required" in error.value.errors


@pytest.fixture
def enrollment_service():
    connection = sqlite3.connect(":memory:")
    repository = EnrollmentRepository(connection)
    repository.create_table()
    return EnrollmentService(repository)


def valid_enrollment():
    return {
        "student_id": 352,
        "course_id": 203,
        "grade": 123.05,
        "completed": False,
    }


def test_create_enrollment_returns_an_id(enrollment_service):
    created = enrollment_service.create_enrollment(valid_enrollment())
    assert created["id"] is not None


def test_get_enrollment_returns_the_created_record(enrollment_service):
    created = enrollment_service.create_enrollment(valid_enrollment())
    fetched = enrollment_service.get_enrollment(created["id"])
    assert fetched["student_id"] == created["student_id"]


def test_get_missing_enrollment_raises(enrollment_service):
    with pytest.raises(NotFoundError):
        enrollment_service.get_enrollment(12345)


def test_list_enrollments_pages(enrollment_service):
    for _ in range(25):
        enrollment_service.create_enrollment(valid_enrollment())
    page = enrollment_service.list_enrollments(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_enrollment(enrollment_service):
    created = enrollment_service.create_enrollment(valid_enrollment())
    enrollment_service.delete_enrollment(created["id"])
    with pytest.raises(NotFoundError):
        enrollment_service.get_enrollment(created["id"])


@pytest.fixture
def room_service():
    connection = sqlite3.connect(":memory:")
    repository = RoomRepository(connection)
    repository.create_table()
    return RoomService(repository)


def valid_room():
    return {
        "number": 'number 1',
        "floor": 333,
        "beds": 383,
        "price_per_night": 889.7,
    }


def test_create_room_returns_an_id(room_service):
    created = room_service.create_room(valid_room())
    assert created["id"] is not None


def test_get_room_returns_the_created_record(room_service):
    created = room_service.create_room(valid_room())
    fetched = room_service.get_room(created["id"])
    assert fetched["number"] == created["number"]


def test_get_missing_room_raises(room_service):
    with pytest.raises(NotFoundError):
        room_service.get_room(12345)


def test_list_rooms_pages(room_service):
    for _ in range(25):
        room_service.create_room(valid_room())
    page = room_service.list_rooms(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_room(room_service):
    created = room_service.create_room(valid_room())
    room_service.delete_room(created["id"])
    with pytest.raises(NotFoundError):
        room_service.get_room(created["id"])


def test_create_room_without_number_is_invalid(room_service):
    data = valid_room()
    data["number"] = ""
    with pytest.raises(ValidationError) as error:
        room_service.create_room(data)
    assert "number is required" in error.value.errors


@pytest.fixture
def booking_service():
    connection = sqlite3.connect(":memory:")
    repository = BookingRepository(connection)
    repository.create_table()
    return BookingService(repository)


def valid_booking():
    return {
        "room_id": 63,
        "guest_name": 'Jack Parker',
        "check_in": '2026-10-04T21:11:00',
        "check_out": '2026-01-06T08:54:00',
        "paid": False,
    }


def test_create_booking_returns_an_id(booking_service):
    created = booking_service.create_booking(valid_booking())
    assert created["id"] is not None


def test_get_booking_returns_the_created_record(booking_service):
    created = booking_service.create_booking(valid_booking())
    fetched = booking_service.get_booking(created["id"])
    assert fetched["room_id"] == created["room_id"]


def test_get_missing_booking_raises(booking_service):
    with pytest.raises(NotFoundError):
        booking_service.get_booking(12345)


def test_list_bookings_pages(booking_service):
    for _ in range(25):
        booking_service.create_booking(valid_booking())
    page = booking_service.list_bookings(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_booking(booking_service):
    created = booking_service.create_booking(valid_booking())
    booking_service.delete_booking(created["id"])
    with pytest.raises(NotFoundError):
        booking_service.get_booking(created["id"])


def test_create_booking_without_guest_name_is_invalid(booking_service):
    data = valid_booking()
    data["guest_name"] = ""
    with pytest.raises(ValidationError) as error:
        booking_service.create_booking(data)
    assert "guest_name is required" in error.value.errors


@pytest.fixture
def subscription_service():
    connection = sqlite3.connect(":memory:")
    repository = SubscriptionRepository(connection)
    repository.create_table()
    return SubscriptionService(repository)


def valid_subscription():
    return {
        "user_id": 324,
        "plan": 'plan 1',
        "price": 493.18,
        "renews_at": '2026-07-08T18:36:00',
        "is_active": False,
    }


def test_create_subscription_returns_an_id(subscription_service):
    created = subscription_service.create_subscription(valid_subscription())
    assert created["id"] is not None


def test_get_subscription_returns_the_created_record(subscription_service):
    created = subscription_service.create_subscription(valid_subscription())
    fetched = subscription_service.get_subscription(created["id"])
    assert fetched["user_id"] == created["user_id"]


def test_get_missing_subscription_raises(subscription_service):
    with pytest.raises(NotFoundError):
        subscription_service.get_subscription(12345)


def test_list_subscriptions_pages(subscription_service):
    for _ in range(25):
        subscription_service.create_subscription(valid_subscription())
    page = subscription_service.list_subscriptions(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_subscription(subscription_service):
    created = subscription_service.create_subscription(valid_subscription())
    subscription_service.delete_subscription(created["id"])
    with pytest.raises(NotFoundError):
        subscription_service.get_subscription(created["id"])


def test_create_subscription_without_plan_is_invalid(subscription_service):
    data = valid_subscription()
    data["plan"] = ""
    with pytest.raises(ValidationError) as error:
        subscription_service.create_subscription(data)
    assert "plan is required" in error.value.errors


@pytest.fixture
def notification_service():
    connection = sqlite3.connect(":memory:")
    repository = NotificationRepository(connection)
    repository.create_table()
    return NotificationService(repository)


def valid_notification():
    return {
        "user_id": 332,
        "title": 'title 1',
        "message": 'message 1',
        "read": False,
        "sent_at": '2026-11-27T08:34:00',
    }


def test_create_notification_returns_an_id(notification_service):
    created = notification_service.create_notification(valid_notification())
    assert created["id"] is not None


def test_get_notification_returns_the_created_record(notification_service):
    created = notification_service.create_notification(valid_notification())
    fetched = notification_service.get_notification(created["id"])
    assert fetched["user_id"] == created["user_id"]


def test_get_missing_notification_raises(notification_service):
    with pytest.raises(NotFoundError):
        notification_service.get_notification(12345)


def test_list_notifications_pages(notification_service):
    for _ in range(25):
        notification_service.create_notification(valid_notification())
    page = notification_service.list_notifications(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_notification(notification_service):
    created = notification_service.create_notification(valid_notification())
    notification_service.delete_notification(created["id"])
    with pytest.raises(NotFoundError):
        notification_service.get_notification(created["id"])


def test_create_notification_without_title_is_invalid(notification_service):
    data = valid_notification()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        notification_service.create_notification(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def session_service():
    connection = sqlite3.connect(":memory:")
    repository = SessionRepository(connection)
    repository.create_table()
    return SessionService(repository)


def valid_session():
    return {
        "user_id": 272,
        "token": 'token 1',
        "expires_at": '2026-10-21T04:32:00',
        "ip_address": 'ip address 1',
    }


def test_create_session_returns_an_id(session_service):
    created = session_service.create_session(valid_session())
    assert created["id"] is not None


def test_get_session_returns_the_created_record(session_service):
    created = session_service.create_session(valid_session())
    fetched = session_service.get_session(created["id"])
    assert fetched["user_id"] == created["user_id"]


def test_get_missing_session_raises(session_service):
    with pytest.raises(NotFoundError):
        session_service.get_session(12345)


def test_list_sessions_pages(session_service):
    for _ in range(25):
        session_service.create_session(valid_session())
    page = session_service.list_sessions(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_session(session_service):
    created = session_service.create_session(valid_session())
    session_service.delete_session(created["id"])
    with pytest.raises(NotFoundError):
        session_service.get_session(created["id"])


def test_create_session_without_token_is_invalid(session_service):
    data = valid_session()
    data["token"] = ""
    with pytest.raises(ValidationError) as error:
        session_service.create_session(data)
    assert "token is required" in error.value.errors


@pytest.fixture
def audit_log_service():
    connection = sqlite3.connect(":memory:")
    repository = AuditLogRepository(connection)
    repository.create_table()
    return AuditLogService(repository)


def valid_audit_log():
    return {
        "actor_id": 76,
        "action": 'action 1',
        "target": 'target 1',
        "created_at": '2026-03-03T04:13:00',
    }


def test_create_audit_log_returns_an_id(audit_log_service):
    created = audit_log_service.create_audit_log(valid_audit_log())
    assert created["id"] is not None


def test_get_audit_log_returns_the_created_record(audit_log_service):
    created = audit_log_service.create_audit_log(valid_audit_log())
    fetched = audit_log_service.get_audit_log(created["id"])
    assert fetched["actor_id"] == created["actor_id"]


def test_get_missing_audit_log_raises(audit_log_service):
    with pytest.raises(NotFoundError):
        audit_log_service.get_audit_log(12345)


def test_list_audit_logs_pages(audit_log_service):
    for _ in range(25):
        audit_log_service.create_audit_log(valid_audit_log())
    page = audit_log_service.list_audit_logs(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_audit_log(audit_log_service):
    created = audit_log_service.create_audit_log(valid_audit_log())
    audit_log_service.delete_audit_log(created["id"])
    with pytest.raises(NotFoundError):
        audit_log_service.get_audit_log(created["id"])


def test_create_audit_log_without_action_is_invalid(audit_log_service):
    data = valid_audit_log()
    data["action"] = ""
    with pytest.raises(ValidationError) as error:
        audit_log_service.create_audit_log(data)
    assert "action is required" in error.value.errors


@pytest.fixture
def setting_service():
    connection = sqlite3.connect(":memory:")
    repository = SettingRepository(connection)
    repository.create_table()
    return SettingService(repository)


def valid_setting():
    return {
        "key": 'key 1',
        "value": 'value 1',
        "description": 'description 1',
    }


def test_create_setting_returns_an_id(setting_service):
    created = setting_service.create_setting(valid_setting())
    assert created["id"] is not None


def test_get_setting_returns_the_created_record(setting_service):
    created = setting_service.create_setting(valid_setting())
    fetched = setting_service.get_setting(created["id"])
    assert fetched["key"] == created["key"]


def test_get_missing_setting_raises(setting_service):
    with pytest.raises(NotFoundError):
        setting_service.get_setting(12345)


def test_list_settings_pages(setting_service):
    for _ in range(25):
        setting_service.create_setting(valid_setting())
    page = setting_service.list_settings(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_setting(setting_service):
    created = setting_service.create_setting(valid_setting())
    setting_service.delete_setting(created["id"])
    with pytest.raises(NotFoundError):
        setting_service.get_setting(created["id"])


def test_create_setting_without_key_is_invalid(setting_service):
    data = valid_setting()
    data["key"] = ""
    with pytest.raises(ValidationError) as error:
        setting_service.create_setting(data)
    assert "key is required" in error.value.errors


@pytest.fixture
def recipe_service():
    connection = sqlite3.connect(":memory:")
    repository = RecipeRepository(connection)
    repository.create_table()
    return RecipeService(repository)


def valid_recipe():
    return {
        "title": 'title 1',
        "servings": 397,
        "prep_minutes": 411,
        "cook_minutes": 432,
        "vegetarian": True,
    }


def test_create_recipe_returns_an_id(recipe_service):
    created = recipe_service.create_recipe(valid_recipe())
    assert created["id"] is not None


def test_get_recipe_returns_the_created_record(recipe_service):
    created = recipe_service.create_recipe(valid_recipe())
    fetched = recipe_service.get_recipe(created["id"])
    assert fetched["title"] == created["title"]


def test_get_missing_recipe_raises(recipe_service):
    with pytest.raises(NotFoundError):
        recipe_service.get_recipe(12345)


def test_list_recipes_pages(recipe_service):
    for _ in range(25):
        recipe_service.create_recipe(valid_recipe())
    page = recipe_service.list_recipes(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_recipe(recipe_service):
    created = recipe_service.create_recipe(valid_recipe())
    recipe_service.delete_recipe(created["id"])
    with pytest.raises(NotFoundError):
        recipe_service.get_recipe(created["id"])


def test_create_recipe_without_title_is_invalid(recipe_service):
    data = valid_recipe()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        recipe_service.create_recipe(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def ingredient_service():
    connection = sqlite3.connect(":memory:")
    repository = IngredientRepository(connection)
    repository.create_table()
    return IngredientService(repository)


def valid_ingredient():
    return {
        "recipe_id": 255,
        "name": 'Chloe Baker',
        "amount": 62.39,
        "unit": 'unit 1',
    }


def test_create_ingredient_returns_an_id(ingredient_service):
    created = ingredient_service.create_ingredient(valid_ingredient())
    assert created["id"] is not None


def test_get_ingredient_returns_the_created_record(ingredient_service):
    created = ingredient_service.create_ingredient(valid_ingredient())
    fetched = ingredient_service.get_ingredient(created["id"])
    assert fetched["recipe_id"] == created["recipe_id"]


def test_get_missing_ingredient_raises(ingredient_service):
    with pytest.raises(NotFoundError):
        ingredient_service.get_ingredient(12345)


def test_list_ingredients_pages(ingredient_service):
    for _ in range(25):
        ingredient_service.create_ingredient(valid_ingredient())
    page = ingredient_service.list_ingredients(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_ingredient(ingredient_service):
    created = ingredient_service.create_ingredient(valid_ingredient())
    ingredient_service.delete_ingredient(created["id"])
    with pytest.raises(NotFoundError):
        ingredient_service.get_ingredient(created["id"])


def test_create_ingredient_without_name_is_invalid(ingredient_service):
    data = valid_ingredient()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        ingredient_service.create_ingredient(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def playlist_service():
    connection = sqlite3.connect(":memory:")
    repository = PlaylistRepository(connection)
    repository.create_table()
    return PlaylistService(repository)


def valid_playlist():
    return {
        "name": 'Felix Turner',
        "owner_id": 292,
        "is_public": False,
        "track_count": 174,
    }


def test_create_playlist_returns_an_id(playlist_service):
    created = playlist_service.create_playlist(valid_playlist())
    assert created["id"] is not None


def test_get_playlist_returns_the_created_record(playlist_service):
    created = playlist_service.create_playlist(valid_playlist())
    fetched = playlist_service.get_playlist(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_playlist_raises(playlist_service):
    with pytest.raises(NotFoundError):
        playlist_service.get_playlist(12345)


def test_list_playlists_pages(playlist_service):
    for _ in range(25):
        playlist_service.create_playlist(valid_playlist())
    page = playlist_service.list_playlists(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_playlist(playlist_service):
    created = playlist_service.create_playlist(valid_playlist())
    playlist_service.delete_playlist(created["id"])
    with pytest.raises(NotFoundError):
        playlist_service.get_playlist(created["id"])


def test_create_playlist_without_name_is_invalid(playlist_service):
    data = valid_playlist()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        playlist_service.create_playlist(data)
    assert "name is required" in error.value.errors


@pytest.fixture
def track_service():
    connection = sqlite3.connect(":memory:")
    repository = TrackRepository(connection)
    repository.create_table()
    return TrackService(repository)


def valid_track():
    return {
        "title": 'title 1',
        "artist": 'artist 1',
        "album": 'album 1',
        "duration_seconds": 221,
    }


def test_create_track_returns_an_id(track_service):
    created = track_service.create_track(valid_track())
    assert created["id"] is not None


def test_get_track_returns_the_created_record(track_service):
    created = track_service.create_track(valid_track())
    fetched = track_service.get_track(created["id"])
    assert fetched["title"] == created["title"]


def test_get_missing_track_raises(track_service):
    with pytest.raises(NotFoundError):
        track_service.get_track(12345)


def test_list_tracks_pages(track_service):
    for _ in range(25):
        track_service.create_track(valid_track())
    page = track_service.list_tracks(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_track(track_service):
    created = track_service.create_track(valid_track())
    track_service.delete_track(created["id"])
    with pytest.raises(NotFoundError):
        track_service.get_track(created["id"])


def test_create_track_without_title_is_invalid(track_service):
    data = valid_track()
    data["title"] = ""
    with pytest.raises(ValidationError) as error:
        track_service.create_track(data)
    assert "title is required" in error.value.errors


@pytest.fixture
def device_service():
    connection = sqlite3.connect(":memory:")
    repository = DeviceRepository(connection)
    repository.create_table()
    return DeviceService(repository)


def valid_device():
    return {
        "name": 'Paula Evans',
        "serial_number": 'serial number 1',
        "firmware": 'firmware 1',
        "online": True,
        "last_seen": '2026-08-20T10:04:00',
    }


def test_create_device_returns_an_id(device_service):
    created = device_service.create_device(valid_device())
    assert created["id"] is not None


def test_get_device_returns_the_created_record(device_service):
    created = device_service.create_device(valid_device())
    fetched = device_service.get_device(created["id"])
    assert fetched["name"] == created["name"]


def test_get_missing_device_raises(device_service):
    with pytest.raises(NotFoundError):
        device_service.get_device(12345)


def test_list_devices_pages(device_service):
    for _ in range(25):
        device_service.create_device(valid_device())
    page = device_service.list_devices(page=2, page_size=10)
    assert len(page["items"]) == 10
    assert page["total"] == 25


def test_delete_device(device_service):
    created = device_service.create_device(valid_device())
    device_service.delete_device(created["id"])
    with pytest.raises(NotFoundError):
        device_service.get_device(created["id"])


def test_create_device_without_name_is_invalid(device_service):
    data = valid_device()
    data["name"] = ""
    with pytest.raises(ValidationError) as error:
        device_service.create_device(data)
    assert "name is required" in error.value.errors

