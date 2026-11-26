-- Schema, sample data and the everyday queries.

BEGIN;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users (email);

INSERT INTO users (username, email, full_name, is_active, created_at) VALUES
    ('username 1', 'jack.johnson1@example.com', 'Rosa Moore', TRUE, '2026-07-18 19:39:00'),
    ('username 2', 'zoe.mitchell2@example.com', 'Quinn Hall', FALSE, '2026-11-01 13:28:00'),
    ('username 3', 'paula.lewis3@example.com', 'Grace Miller', TRUE, '2026-06-08 17:46:00'),
    ('username 4', 'olivia.smith4@example.com', 'Liam Taylor', FALSE, '2026-06-17 15:24:00'),
    ('username 5', 'anna.johnson5@example.com', 'Noah Walker', TRUE, '2026-10-09 06:44:00'),
    ('username 6', 'emma.walker6@example.com', 'Yusuf Miller', TRUE, '2026-01-17 18:25:00'),
    ('username 7', 'quinn.baker7@example.com', 'Felix Evans', FALSE, '2026-02-23 22:17:00'),
    ('username 8', 'zoe.johnson8@example.com', 'Isla Taylor', FALSE, '2026-07-16 01:21:00'),
    ('username 9', 'yusuf.lewis9@example.com', 'Wendy Baker', FALSE, '2026-12-12 10:47:00'),
    ('username 10', 'tara.johnson10@example.com', 'Zoe Lewis', FALSE, '2026-09-23 14:49:00'),
    ('username 11', 'tara.young11@example.com', 'Isla Wright', FALSE, '2026-06-26 12:13:00'),
    ('username 12', 'mia.hall12@example.com', 'Chloe Clark', FALSE, '2026-01-06 19:26:00');

-- All users, newest first.
SELECT * FROM users ORDER BY created_at DESC LIMIT 50;

-- One user by id.
SELECT * FROM users WHERE id = $1;

-- How many users were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM users
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    sku VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description VARCHAR(255) NOT NULL,
    price NUMERIC(12, 2) NOT NULL,
    stock INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_products_sku ON products (sku);

INSERT INTO products (sku, name, description, price, stock) VALUES
    ('sku 1', 'Lucas Hall', 'description 1', 416.27, 355),
    ('sku 2', 'Zoe Lewis', 'description 2', 638.89, 430),
    ('sku 3', 'Zoe Wright', 'description 3', 399.84, 386),
    ('sku 4', 'Olivia Hall', 'description 4', 424.16, 124),
    ('sku 5', 'Tara Walker', 'description 5', 393.62, 456),
    ('sku 6', 'Yusuf Hall', 'description 6', 814.12, 446),
    ('sku 7', 'Yusuf Miller', 'description 7', 805.14, 481),
    ('sku 8', 'Mia Clark', 'description 8', 690.66, 423),
    ('sku 9', 'Nora Turner', 'description 9', 293.02, 476),
    ('sku 10', 'Yusuf Clark', 'description 10', 42.28, 324),
    ('sku 11', 'Henry Miller', 'description 11', 80.89, 44),
    ('sku 12', 'Paula Collins', 'description 12', 446.21, 138);

-- All products, newest first.
SELECT * FROM products ORDER BY created_at DESC LIMIT 50;

-- One product by id.
SELECT * FROM products WHERE id = $1;

-- How many products were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM products
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    status VARCHAR(255) NOT NULL,
    total NUMERIC(12, 2) NOT NULL,
    currency VARCHAR(255) NOT NULL,
    placed_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_orders_customer_id ON orders (customer_id);

INSERT INTO orders (customer_id, status, total, currency, placed_at) VALUES
    (69, 'done', 428.77, 'currency 1', '2026-06-13 13:10:00'),
    (21, 'active', 416.06, 'currency 2', '2026-03-26 17:15:00'),
    (289, 'cancelled', 98.01, 'currency 3', '2026-10-04 07:18:00'),
    (107, 'done', 334.96, 'currency 4', '2026-01-01 22:00:00'),
    (76, 'pending', 746.52, 'currency 5', '2026-11-04 18:29:00'),
    (91, 'cancelled', 782.01, 'currency 6', '2026-12-06 19:42:00'),
    (195, 'active', 32.77, 'currency 7', '2026-04-01 07:33:00'),
    (126, 'active', 908.91, 'currency 8', '2026-11-23 05:58:00'),
    (164, 'cancelled', 51.05, 'currency 9', '2026-11-25 05:13:00'),
    (344, 'pending', 22.94, 'currency 10', '2026-09-17 03:55:00'),
    (61, 'cancelled', 942.06, 'currency 11', '2026-09-05 02:31:00'),
    (132, 'pending', 175.28, 'currency 12', '2026-11-22 16:13:00');

-- All orders, newest first.
SELECT * FROM orders ORDER BY created_at DESC LIMIT 50;

-- One order by id.
SELECT * FROM orders WHERE id = $1;

-- How many orders were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM orders
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_order_items_order_id ON order_items (order_id);
CREATE INDEX idx_order_items_product_id ON order_items (product_id);

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
    (413, 118, 191, 561.97),
    (239, 121, 14, 171.62),
    (281, 306, 248, 611.5),
    (119, 169, 25, 818.42),
    (314, 93, 173, 451.79),
    (147, 217, 84, 881.2),
    (108, 487, 104, 262.16),
    (25, 359, 322, 176.5),
    (110, 190, 69, 330.23),
    (447, 419, 359, 625.09),
    (402, 340, 168, 615.48),
    (439, 386, 222, 873.18);

-- All order items, newest first.
SELECT * FROM order_items ORDER BY created_at DESC LIMIT 50;

-- One order item by id.
SELECT * FROM order_items WHERE id = $1;

-- How many order items were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM order_items
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(255) NOT NULL,
    country VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_customers_email ON customers (email);

INSERT INTO customers (first_name, last_name, email, phone, country) VALUES
    ('Isla', 'Carter', 'mia.smith1@example.com', '+49 30 4962711', 'FR'),
    ('Lucas', 'Lewis', 'zoe.wilson2@example.com', '+49 30 7908592', 'ES'),
    ('Tara', 'Collins', 'lucas.collins3@example.com', '+49 30 4770809', 'ES'),
    ('David', 'Taylor', 'noah.lewis4@example.com', '+49 30 4998424', 'DE'),
    ('David', 'Evans', 'tara.adams5@example.com', '+49 30 8522082', 'PT'),
    ('Noah', 'Walker', 'zoe.miller6@example.com', '+49 30 7099925', 'SE'),
    ('Anna', 'Smith', 'lucas.hall7@example.com', '+49 30 8941980', 'NL'),
    ('Olivia', 'Clark', 'ben.king8@example.com', '+49 30 2670816', 'SE'),
    ('Lucas', 'Collins', 'victor.carter9@example.com', '+49 30 4485313', 'PL'),
    ('Paula', 'Hall', 'henry.lewis10@example.com', '+49 30 8276071', 'AT'),
    ('Jack', 'Johnson', 'quinn.lewis11@example.com', '+49 30 1794684', 'NL'),
    ('Mark', 'Hall', 'emma.wright12@example.com', '+49 30 3184275', 'AT');

-- All customers, newest first.
SELECT * FROM customers ORDER BY created_at DESC LIMIT 50;

-- One customer by id.
SELECT * FROM customers WHERE id = $1;

-- How many customers were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM customers
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE invoices (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    number VARCHAR(255) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL,
    due_date TIMESTAMP NOT NULL,
    paid BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_invoices_order_id ON invoices (order_id);

INSERT INTO invoices (order_id, number, amount, due_date, paid) VALUES
    (330, 'number 1', 78.42, '2026-03-04 11:14:00', FALSE),
    (194, 'number 2', 659.53, '2026-03-21 05:06:00', FALSE),
    (299, 'number 3', 349.48, '2026-06-08 02:03:00', FALSE),
    (437, 'number 4', 753.94, '2026-04-10 03:31:00', FALSE),
    (258, 'number 5', 706.73, '2026-10-23 08:29:00', TRUE),
    (406, 'number 6', 589.72, '2026-05-25 20:01:00', TRUE),
    (168, 'number 7', 876.62, '2026-02-20 04:23:00', TRUE),
    (386, 'number 8', 787.54, '2026-08-23 08:07:00', FALSE),
    (491, 'number 9', 953.38, '2026-05-07 12:19:00', TRUE),
    (141, 'number 10', 902.8, '2026-11-02 11:57:00', FALSE),
    (229, 'number 11', 700.62, '2026-01-17 05:29:00', TRUE),
    (79, 'number 12', 368.24, '2026-02-07 10:43:00', TRUE);

-- All invoices, newest first.
SELECT * FROM invoices ORDER BY created_at DESC LIMIT 50;

-- One invoice by id.
SELECT * FROM invoices WHERE id = $1;

-- How many invoices were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM invoices
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE payments (
    id SERIAL PRIMARY KEY,
    invoice_id INTEGER NOT NULL,
    method VARCHAR(255) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL,
    reference VARCHAR(255) NOT NULL,
    received_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_payments_invoice_id ON payments (invoice_id);

INSERT INTO payments (invoice_id, method, amount, reference, received_at) VALUES
    (336, 'method 1', 907.31, 'reference 1', '2026-04-08 11:04:00'),
    (368, 'method 2', 595.65, 'reference 2', '2026-08-01 11:54:00'),
    (26, 'method 3', 998.71, 'reference 3', '2026-09-21 16:28:00'),
    (16, 'method 4', 195.72, 'reference 4', '2026-12-23 16:27:00'),
    (498, 'method 5', 368.88, 'reference 5', '2026-02-26 21:29:00'),
    (363, 'method 6', 32.36, 'reference 6', '2026-06-18 14:16:00'),
    (72, 'method 7', 126.81, 'reference 7', '2026-12-12 22:26:00'),
    (469, 'method 8', 529.38, 'reference 8', '2026-06-24 19:25:00'),
    (153, 'method 9', 731.69, 'reference 9', '2026-02-05 23:50:00'),
    (290, 'method 10', 743.84, 'reference 10', '2026-09-06 12:02:00'),
    (408, 'method 11', 319.05, 'reference 11', '2026-07-24 09:32:00'),
    (189, 'method 12', 562.32, 'reference 12', '2026-05-24 16:17:00');

-- All payments, newest first.
SELECT * FROM payments ORDER BY created_at DESC LIMIT 50;

-- One payment by id.
SELECT * FROM payments WHERE id = $1;

-- How many payments were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM payments
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL,
    parent_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_slug ON categories (slug);
CREATE INDEX idx_categories_parent_id ON categories (parent_id);

INSERT INTO categories (name, slug, parent_id, position) VALUES
    ('Zoe Brown', 'slug 1', 379, 309),
    ('Sam Parker', 'slug 2', 181, 394),
    ('Lucas Evans', 'slug 3', 109, 162),
    ('Tara Evans', 'slug 4', 232, 285),
    ('Jack Smith', 'slug 5', 283, 479),
    ('Oscar Wright', 'slug 6', 248, 204),
    ('David Green', 'slug 7', 259, 239),
    ('Paula Johnson', 'slug 8', 224, 112),
    ('Lena Adams', 'slug 9', 493, 260),
    ('Emma Lewis', 'slug 10', 484, 152),
    ('Victor Young', 'slug 11', 110, 113),
    ('Emma Adams', 'slug 12', 114, 75);

-- All categories, newest first.
SELECT * FROM categories ORDER BY created_at DESC LIMIT 50;

-- One category by id.
SELECT * FROM categories WHERE id = $1;

-- How many categories were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM categories
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    body VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reviews_product_id ON reviews (product_id);
CREATE INDEX idx_reviews_user_id ON reviews (user_id);

INSERT INTO reviews (product_id, user_id, rating, title, body) VALUES
    (394, 107, 225, 'title 1', 'body 1'),
    (303, 326, 2, 'title 2', 'body 2'),
    (308, 282, 214, 'title 3', 'body 3'),
    (253, 162, 298, 'title 4', 'body 4'),
    (345, 333, 463, 'title 5', 'body 5'),
    (255, 30, 6, 'title 6', 'body 6'),
    (269, 5, 45, 'title 7', 'body 7'),
    (311, 268, 299, 'title 8', 'body 8'),
    (175, 237, 45, 'title 9', 'body 9'),
    (367, 68, 55, 'title 10', 'body 10'),
    (180, 156, 473, 'title 11', 'body 11'),
    (150, 198, 300, 'title 12', 'body 12');

-- All reviews, newest first.
SELECT * FROM reviews ORDER BY created_at DESC LIMIT 50;

-- One review by id.
SELECT * FROM reviews WHERE id = $1;

-- How many reviews were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM reviews
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE addresses (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    street VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL,
    postal_code VARCHAR(255) NOT NULL,
    country VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_addresses_customer_id ON addresses (customer_id);

INSERT INTO addresses (customer_id, street, city, postal_code, country) VALUES
    (372, 'street 1', 'Warsaw', 'postal code 1', 'SE'),
    (344, 'street 2', 'Milan', 'postal code 2', 'FR'),
    (124, 'street 3', 'Dublin', 'postal code 3', 'AT'),
    (363, 'street 4', 'Prague', 'postal code 4', 'ES'),
    (156, 'street 5', 'Lisbon', 'postal code 5', 'PT'),
    (383, 'street 6', 'Copenhagen', 'postal code 6', 'AT'),
    (359, 'street 7', 'Brussels', 'postal code 7', 'PT'),
    (398, 'street 8', 'Rome', 'postal code 8', 'ES'),
    (301, 'street 9', 'Lisbon', 'postal code 9', 'PL'),
    (278, 'street 10', 'Warsaw', 'postal code 10', 'DE'),
    (187, 'street 11', 'Paris', 'postal code 11', 'IE'),
    (159, 'street 12', 'Milan', 'postal code 12', 'AT');

-- All addresses, newest first.
SELECT * FROM addresses ORDER BY created_at DESC LIMIT 50;

-- One address by id.
SELECT * FROM addresses WHERE id = $1;

-- How many addresses were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM addresses
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE shipments (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    carrier VARCHAR(255) NOT NULL,
    tracking_number VARCHAR(255) NOT NULL,
    shipped_at TIMESTAMP NOT NULL,
    delivered BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_shipments_order_id ON shipments (order_id);

INSERT INTO shipments (order_id, carrier, tracking_number, shipped_at, delivered) VALUES
    (382, 'carrier 1', 'tracking number 1', '2026-11-28 21:52:00', TRUE),
    (59, 'carrier 2', 'tracking number 2', '2026-12-21 22:03:00', FALSE),
    (10, 'carrier 3', 'tracking number 3', '2026-07-20 00:58:00', FALSE),
    (446, 'carrier 4', 'tracking number 4', '2026-02-01 15:06:00', TRUE),
    (188, 'carrier 5', 'tracking number 5', '2026-03-26 21:36:00', TRUE),
    (299, 'carrier 6', 'tracking number 6', '2026-10-24 07:18:00', FALSE),
    (48, 'carrier 7', 'tracking number 7', '2026-06-22 08:57:00', TRUE),
    (15, 'carrier 8', 'tracking number 8', '2026-07-01 05:47:00', TRUE),
    (439, 'carrier 9', 'tracking number 9', '2026-03-18 15:58:00', FALSE),
    (346, 'carrier 10', 'tracking number 10', '2026-09-14 12:01:00', TRUE),
    (230, 'carrier 11', 'tracking number 11', '2026-06-06 09:57:00', FALSE),
    (293, 'carrier 12', 'tracking number 12', '2026-12-04 16:53:00', TRUE);

-- All shipments, newest first.
SELECT * FROM shipments ORDER BY created_at DESC LIMIT 50;

-- One shipment by id.
SELECT * FROM shipments WHERE id = $1;

-- How many shipments were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM shipments
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE warehouses (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL,
    capacity INTEGER NOT NULL,
    is_open BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO warehouses (name, city, capacity, is_open) VALUES
    ('Henry Taylor', 'Lisbon', 358, FALSE),
    ('David King', 'Copenhagen', 407, FALSE),
    ('Nora Johnson', 'Milan', 390, TRUE),
    ('Liam Nelson', 'Budapest', 240, TRUE),
    ('Olivia King', 'Brussels', 71, TRUE),
    ('Isla Miller', 'Munich', 432, FALSE),
    ('Lena Brown', 'Lisbon', 156, FALSE),
    ('Mia Wright', 'Rome', 62, FALSE),
    ('Mark Smith', 'Lisbon', 342, TRUE),
    ('Rosa Young', 'Athens', 176, FALSE),
    ('Quinn Baker', 'Amsterdam', 81, FALSE),
    ('Emma Smith', 'Oslo', 146, FALSE);

-- All warehouses, newest first.
SELECT * FROM warehouses ORDER BY created_at DESC LIMIT 50;

-- One warehouse by id.
SELECT * FROM warehouses WHERE id = $1;

-- How many warehouses were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM warehouses
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    contact_email VARCHAR(255) NOT NULL,
    phone VARCHAR(255) NOT NULL,
    country VARCHAR(255) NOT NULL,
    rating INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO suppliers (name, contact_email, phone, country, rating) VALUES
    ('Felix Taylor', 'noah.baker1@example.com', '+49 30 1451149', 'NL', 157),
    ('Emma Baker', 'liam.carter2@example.com', '+49 30 4988264', 'IE', 240),
    ('Henry Young', 'jack.adams3@example.com', '+49 30 2518666', 'SE', 104),
    ('Henry Moore', 'victor.taylor4@example.com', '+49 30 8266557', 'PL', 13),
    ('Sam Wilson', 'anna.moore5@example.com', '+49 30 3503508', 'AT', 457),
    ('Liam Wright', 'zoe.smith6@example.com', '+49 30 7321944', 'NL', 374),
    ('Tara Lewis', 'victor.johnson7@example.com', '+49 30 6520791', 'PT', 50),
    ('Rosa Smith', 'isla.lewis8@example.com', '+49 30 2571692', 'AT', 92),
    ('Olivia Lewis', 'david.collins9@example.com', '+49 30 5283833', 'FR', 464),
    ('Anna Lewis', 'rosa.miller10@example.com', '+49 30 8622895', 'ES', 396),
    ('Ben Brown', 'ben.taylor11@example.com', '+49 30 9519977', 'IE', 384),
    ('Isla Wright', 'isla.mitchell12@example.com', '+49 30 3388821', 'DE', 152);

-- All suppliers, newest first.
SELECT * FROM suppliers ORDER BY created_at DESC LIMIT 50;

-- One supplier by id.
SELECT * FROM suppliers WHERE id = $1;

-- How many suppliers were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM suppliers
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE coupons (
    id SERIAL PRIMARY KEY,
    code VARCHAR(255) NOT NULL,
    discount_percent INTEGER NOT NULL,
    valid_until TIMESTAMP NOT NULL,
    max_uses INTEGER NOT NULL,
    is_active BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_coupons_code ON coupons (code);

INSERT INTO coupons (code, discount_percent, valid_until, max_uses, is_active) VALUES
    ('code 1', 340, '2026-05-25 18:29:00', 219, TRUE),
    ('code 2', 446, '2026-08-01 16:33:00', 192, TRUE),
    ('code 3', 161, '2026-04-19 21:25:00', 374, TRUE),
    ('code 4', 20, '2026-10-04 21:15:00', 132, FALSE),
    ('code 5', 123, '2026-12-07 04:41:00', 464, FALSE),
    ('code 6', 495, '2026-10-18 10:19:00', 127, FALSE),
    ('code 7', 274, '2026-05-12 04:51:00', 217, FALSE),
    ('code 8', 113, '2026-06-15 14:35:00', 371, TRUE),
    ('code 9', 275, '2026-10-28 17:59:00', 186, FALSE),
    ('code 10', 330, '2026-09-21 01:00:00', 195, TRUE),
    ('code 11', 36, '2026-07-07 13:51:00', 101, TRUE),
    ('code 12', 378, '2026-09-11 09:48:00', 242, FALSE);

-- All coupons, newest first.
SELECT * FROM coupons ORDER BY created_at DESC LIMIT 50;

-- One coupon by id.
SELECT * FROM coupons WHERE id = $1;

-- How many coupons were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM coupons
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    department VARCHAR(255) NOT NULL,
    salary NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_employees_email ON employees (email);

INSERT INTO employees (first_name, last_name, email, department, salary) VALUES
    ('Yusuf', 'Smith', 'chloe.miller1@example.com', 'department 1', 221.99),
    ('Zoe', 'Taylor', 'ben.smith2@example.com', 'department 2', 205.32),
    ('Henry', 'Adams', 'jack.wright3@example.com', 'department 3', 174.29),
    ('Emma', 'Mitchell', 'liam.johnson4@example.com', 'department 4', 48.71),
    ('Mark', 'Nelson', 'mark.lewis5@example.com', 'department 5', 200.54),
    ('Victor', 'Mitchell', 'nora.collins6@example.com', 'department 6', 405.93),
    ('Isla', 'Collins', 'paula.hall7@example.com', 'department 7', 925.22),
    ('Sam', 'Moore', 'henry.hall8@example.com', 'department 8', 554.46),
    ('Yusuf', 'Brown', 'lena.evans9@example.com', 'department 9', 888.03),
    ('Anna', 'Collins', 'ben.wright10@example.com', 'department 10', 273.23),
    ('Ben', 'Walker', 'grace.adams11@example.com', 'department 11', 561.74),
    ('Isla', 'Brown', 'grace.moore12@example.com', 'department 12', 952.17);

-- All employees, newest first.
SELECT * FROM employees ORDER BY created_at DESC LIMIT 50;

-- One employee by id.
SELECT * FROM employees WHERE id = $1;

-- How many employees were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM employees
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    code VARCHAR(255) NOT NULL,
    manager_id INTEGER NOT NULL,
    budget NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_departments_code ON departments (code);
CREATE INDEX idx_departments_manager_id ON departments (manager_id);

INSERT INTO departments (name, code, manager_id, budget) VALUES
    ('Tara Nelson', 'code 1', 334, 495.47),
    ('Henry Mitchell', 'code 2', 246, 516.58),
    ('Nora Mitchell', 'code 3', 158, 442.24),
    ('Yusuf Wright', 'code 4', 250, 129.04),
    ('Henry Carter', 'code 5', 453, 490.08),
    ('Zoe Wilson', 'code 6', 34, 559.89),
    ('Zoe Green', 'code 7', 23, 626.87),
    ('Tara Nelson', 'code 8', 366, 825.0),
    ('Emma Smith', 'code 9', 388, 869.36),
    ('Anna Smith', 'code 10', 320, 296.16),
    ('Olivia Turner', 'code 11', 25, 478.08),
    ('Rosa Clark', 'code 12', 24, 828.41);

-- All departments, newest first.
SELECT * FROM departments ORDER BY created_at DESC LIMIT 50;

-- One department by id.
SELECT * FROM departments WHERE id = $1;

-- How many departments were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM departments
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description VARCHAR(255) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    budget NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO projects (name, description, start_date, end_date, budget) VALUES
    ('Olivia Johnson', 'description 1', '2026-07-03 16:04:00', '2026-02-17 05:25:00', 106.67),
    ('David Collins', 'description 2', '2026-07-11 07:47:00', '2026-06-04 06:59:00', 520.45),
    ('Mia Moore', 'description 3', '2026-08-08 15:17:00', '2026-11-01 07:55:00', 57.83),
    ('Mark Taylor', 'description 4', '2026-09-27 09:32:00', '2026-06-24 17:02:00', 595.69),
    ('Oscar Moore', 'description 5', '2026-07-20 09:54:00', '2026-09-28 23:03:00', 779.57),
    ('Oscar Clark', 'description 6', '2026-04-18 08:27:00', '2026-12-25 23:31:00', 998.59),
    ('Sam Turner', 'description 7', '2026-06-25 14:16:00', '2026-03-09 23:31:00', 228.12),
    ('Oscar Collins', 'description 8', '2026-07-28 03:13:00', '2026-07-04 10:11:00', 180.94),
    ('Rosa Green', 'description 9', '2026-03-05 02:46:00', '2026-07-10 06:02:00', 887.3),
    ('Yusuf King', 'description 10', '2026-03-15 20:33:00', '2026-11-14 17:46:00', 364.05),
    ('Oscar Mitchell', 'description 11', '2026-05-26 14:58:00', '2026-05-19 23:31:00', 783.51),
    ('Wendy Green', 'description 12', '2026-10-17 08:21:00', '2026-09-26 01:41:00', 935.29);

-- All projects, newest first.
SELECT * FROM projects ORDER BY created_at DESC LIMIT 50;

-- One project by id.
SELECT * FROM projects WHERE id = $1;

-- How many projects were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM projects
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    status VARCHAR(255) NOT NULL,
    priority INTEGER NOT NULL,
    due_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tasks_project_id ON tasks (project_id);

INSERT INTO tasks (project_id, title, status, priority, due_date) VALUES
    (376, 'title 1', 'active', 425, '2026-09-16 00:04:00'),
    (266, 'title 2', 'cancelled', 68, '2026-04-06 18:11:00'),
    (425, 'title 3', 'done', 239, '2026-08-06 22:53:00'),
    (491, 'title 4', 'active', 286, '2026-01-15 22:09:00'),
    (56, 'title 5', 'active', 164, '2026-01-25 10:07:00'),
    (430, 'title 6', 'active', 253, '2026-04-20 11:59:00'),
    (473, 'title 7', 'done', 66, '2026-12-16 17:20:00'),
    (191, 'title 8', 'done', 15, '2026-09-21 05:12:00'),
    (494, 'title 9', 'done', 488, '2026-04-01 12:04:00'),
    (186, 'title 10', 'cancelled', 438, '2026-06-03 08:51:00'),
    (154, 'title 11', 'pending', 139, '2026-01-27 00:02:00'),
    (26, 'title 12', 'done', 213, '2026-08-18 19:02:00');

-- All tasks, newest first.
SELECT * FROM tasks ORDER BY created_at DESC LIMIT 50;

-- One task by id.
SELECT * FROM tasks WHERE id = $1;

-- How many tasks were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM tasks
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    body VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_comments_task_id ON comments (task_id);
CREATE INDEX idx_comments_author_id ON comments (author_id);

INSERT INTO comments (task_id, author_id, body, created_at) VALUES
    (131, 29, 'body 1', '2026-09-26 02:55:00'),
    (389, 2, 'body 2', '2026-03-12 19:13:00'),
    (375, 367, 'body 3', '2026-02-22 04:09:00'),
    (468, 332, 'body 4', '2026-01-15 00:50:00'),
    (271, 405, 'body 5', '2026-12-21 12:29:00'),
    (318, 173, 'body 6', '2026-01-15 15:17:00'),
    (388, 487, 'body 7', '2026-07-23 22:13:00'),
    (16, 410, 'body 8', '2026-11-27 03:31:00'),
    (96, 463, 'body 9', '2026-03-12 07:28:00'),
    (260, 9, 'body 10', '2026-03-06 15:24:00'),
    (322, 379, 'body 11', '2026-06-28 01:39:00'),
    (203, 42, 'body 12', '2026-09-08 22:34:00');

-- All comments, newest first.
SELECT * FROM comments ORDER BY created_at DESC LIMIT 50;

-- One comment by id.
SELECT * FROM comments WHERE id = $1;

-- How many comments were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM comments
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    color VARCHAR(255) NOT NULL,
    usage_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO tags (name, color, usage_count) VALUES
    ('Henry Wright', 'color 1', 303),
    ('Mark Taylor', 'color 2', 423),
    ('Emma Hall', 'color 3', 319),
    ('Felix Baker', 'color 4', 461),
    ('Oscar Turner', 'color 5', 229),
    ('David Moore', 'color 6', 358),
    ('Emma Mitchell', 'color 7', 98),
    ('Tara Taylor', 'color 8', 120),
    ('Quinn Turner', 'color 9', 96),
    ('Wendy Green', 'color 10', 181),
    ('Jack Miller', 'color 11', 438),
    ('Tara Turner', 'color 12', 270);

-- All tags, newest first.
SELECT * FROM tags ORDER BY created_at DESC LIMIT 50;

-- One tag by id.
SELECT * FROM tags WHERE id = $1;

-- How many tags were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM tags
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL,
    body VARCHAR(255) NOT NULL,
    author_id INTEGER NOT NULL,
    published BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_articles_slug ON articles (slug);
CREATE INDEX idx_articles_author_id ON articles (author_id);

INSERT INTO articles (title, slug, body, author_id, published) VALUES
    ('title 1', 'slug 1', 'body 1', 430, FALSE),
    ('title 2', 'slug 2', 'body 2', 498, TRUE),
    ('title 3', 'slug 3', 'body 3', 90, FALSE),
    ('title 4', 'slug 4', 'body 4', 410, TRUE),
    ('title 5', 'slug 5', 'body 5', 402, TRUE),
    ('title 6', 'slug 6', 'body 6', 132, TRUE),
    ('title 7', 'slug 7', 'body 7', 353, FALSE),
    ('title 8', 'slug 8', 'body 8', 207, TRUE),
    ('title 9', 'slug 9', 'body 9', 338, TRUE),
    ('title 10', 'slug 10', 'body 10', 419, TRUE),
    ('title 11', 'slug 11', 'body 11', 276, FALSE),
    ('title 12', 'slug 12', 'body 12', 439, TRUE);

-- All articles, newest first.
SELECT * FROM articles ORDER BY created_at DESC LIMIT 50;

-- One article by id.
SELECT * FROM articles WHERE id = $1;

-- How many articles were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM articles
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE authors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    bio VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    website VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_authors_email ON authors (email);

INSERT INTO authors (name, bio, email, website) VALUES
    ('Isla King', 'bio 1', 'jack.johnson1@example.com', 'website 1'),
    ('Rosa Green', 'bio 2', 'noah.young2@example.com', 'website 2'),
    ('Zoe Mitchell', 'bio 3', 'ben.clark3@example.com', 'website 3'),
    ('Noah Baker', 'bio 4', 'yusuf.young4@example.com', 'website 4'),
    ('Noah Carter', 'bio 5', 'isla.carter5@example.com', 'website 5'),
    ('Isla Parker', 'bio 6', 'zoe.taylor6@example.com', 'website 6'),
    ('Rosa Wilson', 'bio 7', 'nora.green7@example.com', 'website 7'),
    ('David Clark', 'bio 8', 'wendy.lewis8@example.com', 'website 8'),
    ('Isla Mitchell', 'bio 9', 'ben.turner9@example.com', 'website 9'),
    ('Victor Brown', 'bio 10', 'mark.moore10@example.com', 'website 10'),
    ('Quinn Taylor', 'bio 11', 'chloe.smith11@example.com', 'website 11'),
    ('Victor Wright', 'bio 12', 'quinn.johnson12@example.com', 'website 12');

-- All authors, newest first.
SELECT * FROM authors ORDER BY created_at DESC LIMIT 50;

-- One author by id.
SELECT * FROM authors WHERE id = $1;

-- How many authors were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM authors
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    isbn VARCHAR(255) NOT NULL,
    author_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    pages INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_books_author_id ON books (author_id);

INSERT INTO books (title, isbn, author_id, year, pages) VALUES
    ('title 1', 'isbn 1', 360, 280, 207),
    ('title 2', 'isbn 2', 335, 407, 212),
    ('title 3', 'isbn 3', 53, 130, 74),
    ('title 4', 'isbn 4', 198, 350, 107),
    ('title 5', 'isbn 5', 21, 73, 496),
    ('title 6', 'isbn 6', 476, 385, 176),
    ('title 7', 'isbn 7', 54, 204, 428),
    ('title 8', 'isbn 8', 12, 46, 486),
    ('title 9', 'isbn 9', 115, 495, 213),
    ('title 10', 'isbn 10', 298, 315, 402),
    ('title 11', 'isbn 11', 28, 242, 158),
    ('title 12', 'isbn 12', 130, 9, 303);

-- All books, newest first.
SELECT * FROM books ORDER BY created_at DESC LIMIT 50;

-- One book by id.
SELECT * FROM books WHERE id = $1;

-- How many books were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM books
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE libraries (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL,
    opening_hour INTEGER NOT NULL,
    closing_hour INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO libraries (name, city, opening_hour, closing_hour) VALUES
    ('Paula Moore', 'Oslo', 357, 442),
    ('Emma Turner', 'Brussels', 431, 446),
    ('Tara Lewis', 'Warsaw', 402, 273),
    ('Sam Walker', 'Amsterdam', 308, 74),
    ('Rosa Nelson', 'Zurich', 454, 94),
    ('Emma Hall', 'Warsaw', 272, 338),
    ('Grace Hall', 'Oslo', 169, 208),
    ('Oscar Hall', 'Rome', 254, 339),
    ('Zoe Adams', 'Athens', 186, 283),
    ('Felix Nelson', 'Warsaw', 97, 367),
    ('Rosa Wilson', 'Brussels', 415, 30),
    ('David Lewis', 'Lisbon', 313, 255);

-- All libraries, newest first.
SELECT * FROM libraries ORDER BY created_at DESC LIMIT 50;

-- One library by id.
SELECT * FROM libraries WHERE id = $1;

-- How many libraries were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM libraries
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE loans (
    id SERIAL PRIMARY KEY,
    book_id INTEGER NOT NULL,
    member_id INTEGER NOT NULL,
    borrowed_at TIMESTAMP NOT NULL,
    due_date TIMESTAMP NOT NULL,
    returned BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_loans_book_id ON loans (book_id);
CREATE INDEX idx_loans_member_id ON loans (member_id);

INSERT INTO loans (book_id, member_id, borrowed_at, due_date, returned) VALUES
    (204, 431, '2026-09-26 04:06:00', '2026-09-19 09:22:00', TRUE),
    (175, 231, '2026-08-22 20:23:00', '2026-02-03 09:48:00', FALSE),
    (224, 212, '2026-03-20 17:27:00', '2026-01-25 15:46:00', FALSE),
    (278, 493, '2026-11-18 18:08:00', '2026-03-11 01:21:00', TRUE),
    (401, 173, '2026-01-26 12:37:00', '2026-08-07 05:05:00', TRUE),
    (237, 154, '2026-05-25 18:15:00', '2026-07-28 17:04:00', TRUE),
    (75, 267, '2026-09-23 17:56:00', '2026-07-18 19:00:00', FALSE),
    (2, 21, '2026-02-09 04:26:00', '2026-08-01 09:14:00', FALSE),
    (293, 210, '2026-10-13 20:26:00', '2026-06-05 07:45:00', FALSE),
    (403, 151, '2026-04-24 08:14:00', '2026-05-02 05:52:00', TRUE),
    (461, 228, '2026-12-10 16:01:00', '2026-10-04 06:00:00', FALSE),
    (359, 387, '2026-06-03 03:20:00', '2026-11-08 18:39:00', TRUE);

-- All loans, newest first.
SELECT * FROM loans ORDER BY created_at DESC LIMIT 50;

-- One loan by id.
SELECT * FROM loans WHERE id = $1;

-- How many loans were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM loans
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE members (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    joined_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_members_email ON members (email);

INSERT INTO members (first_name, last_name, email, joined_at) VALUES
    ('Jack', 'Moore', 'grace.baker1@example.com', '2026-01-13 09:23:00'),
    ('Anna', 'Carter', 'olivia.young2@example.com', '2026-09-11 07:26:00'),
    ('Wendy', 'Turner', 'nora.parker3@example.com', '2026-07-23 10:48:00'),
    ('Oscar', 'Wilson', 'anna.hall4@example.com', '2026-09-13 20:18:00'),
    ('Zoe', 'Wright', 'tara.mitchell5@example.com', '2026-12-15 22:44:00'),
    ('Felix', 'King', 'isla.turner6@example.com', '2026-01-22 03:35:00'),
    ('Ben', 'Walker', 'wendy.young7@example.com', '2026-08-22 22:13:00'),
    ('Liam', 'Johnson', 'victor.king8@example.com', '2026-05-02 12:04:00'),
    ('Mia', 'Adams', 'quinn.king9@example.com', '2026-10-16 23:40:00'),
    ('Noah', 'Clark', 'rosa.adams10@example.com', '2026-10-15 07:34:00'),
    ('Sam', 'Clark', 'noah.smith11@example.com', '2026-11-26 18:41:00'),
    ('Nora', 'Wright', 'rosa.smith12@example.com', '2026-03-05 19:08:00');

-- All members, newest first.
SELECT * FROM members ORDER BY created_at DESC LIMIT 50;

-- One member by id.
SELECT * FROM members WHERE id = $1;

-- How many members were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM members
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    starts_at TIMESTAMP NOT NULL,
    ends_at TIMESTAMP NOT NULL,
    capacity INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO events (title, location, starts_at, ends_at, capacity) VALUES
    ('title 1', 'location 1', '2026-04-25 03:06:00', '2026-05-21 09:04:00', 359),
    ('title 2', 'location 2', '2026-05-04 20:58:00', '2026-07-12 16:18:00', 79),
    ('title 3', 'location 3', '2026-04-15 05:48:00', '2026-02-03 03:55:00', 295),
    ('title 4', 'location 4', '2026-05-25 17:26:00', '2026-05-12 19:01:00', 434),
    ('title 5', 'location 5', '2026-02-18 20:59:00', '2026-12-18 07:29:00', 200),
    ('title 6', 'location 6', '2026-03-23 07:23:00', '2026-11-06 07:22:00', 21),
    ('title 7', 'location 7', '2026-04-01 20:34:00', '2026-03-04 13:45:00', 341),
    ('title 8', 'location 8', '2026-08-04 06:43:00', '2026-12-08 04:33:00', 202),
    ('title 9', 'location 9', '2026-11-18 01:08:00', '2026-05-12 08:23:00', 45),
    ('title 10', 'location 10', '2026-01-23 05:43:00', '2026-10-26 08:19:00', 354),
    ('title 11', 'location 11', '2026-06-01 06:44:00', '2026-03-16 07:38:00', 125),
    ('title 12', 'location 12', '2026-10-08 05:42:00', '2026-04-07 02:40:00', 122);

-- All events, newest first.
SELECT * FROM events ORDER BY created_at DESC LIMIT 50;

-- One event by id.
SELECT * FROM events WHERE id = $1;

-- How many events were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM events
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE tickets (
    id SERIAL PRIMARY KEY,
    event_id INTEGER NOT NULL,
    holder_name VARCHAR(255) NOT NULL,
    seat VARCHAR(255) NOT NULL,
    price NUMERIC(12, 2) NOT NULL,
    checked_in BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tickets_event_id ON tickets (event_id);

INSERT INTO tickets (event_id, holder_name, seat, price, checked_in) VALUES
    (187, 'Yusuf Johnson', 'seat 1', 885.83, TRUE),
    (5, 'Quinn Wright', 'seat 2', 876.23, FALSE),
    (2, 'Nora Mitchell', 'seat 3', 583.37, TRUE),
    (238, 'Paula Evans', 'seat 4', 936.84, FALSE),
    (264, 'Olivia Taylor', 'seat 5', 800.42, FALSE),
    (486, 'Tara Johnson', 'seat 6', 191.69, FALSE),
    (30, 'Chloe Wilson', 'seat 7', 690.44, FALSE),
    (349, 'Liam Nelson', 'seat 8', 777.73, TRUE),
    (425, 'Rosa Smith', 'seat 9', 59.48, TRUE),
    (197, 'Liam King', 'seat 10', 53.42, FALSE),
    (208, 'Anna Adams', 'seat 11', 624.51, FALSE),
    (423, 'Mark Taylor', 'seat 12', 506.34, FALSE);

-- All tickets, newest first.
SELECT * FROM tickets ORDER BY created_at DESC LIMIT 50;

-- One ticket by id.
SELECT * FROM tickets WHERE id = $1;

-- How many tickets were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM tickets
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE venues (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL,
    capacity INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO venues (name, address, city, capacity) VALUES
    ('Tara Green', 'address 1', 'Stockholm', 226),
    ('Victor Mitchell', 'address 2', 'Paris', 45),
    ('David Hall', 'address 3', 'Berlin', 388),
    ('Lena Green', 'address 4', 'Prague', 22),
    ('Victor Green', 'address 5', 'Brussels', 121),
    ('Mark Mitchell', 'address 6', 'Munich', 63),
    ('Grace Hall', 'address 7', 'Zurich', 164),
    ('Oscar Collins', 'address 8', 'Oslo', 280),
    ('Mark Smith', 'address 9', 'Paris', 134),
    ('Zoe Adams', 'address 10', 'Milan', 328),
    ('Sam Moore', 'address 11', 'Vienna', 451),
    ('Sam Young', 'address 12', 'Oslo', 188);

-- All venues, newest first.
SELECT * FROM venues ORDER BY created_at DESC LIMIT 50;

-- One venue by id.
SELECT * FROM venues WHERE id = $1;

-- How many venues were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM venues
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE vehicles (
    id SERIAL PRIMARY KEY,
    plate VARCHAR(255) NOT NULL,
    make VARCHAR(255) NOT NULL,
    model VARCHAR(255) NOT NULL,
    year INTEGER NOT NULL,
    mileage INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO vehicles (plate, make, model, year, mileage) VALUES
    ('plate 1', 'make 1', 'model 1', 376, 500),
    ('plate 2', 'make 2', 'model 2', 75, 329),
    ('plate 3', 'make 3', 'model 3', 257, 37),
    ('plate 4', 'make 4', 'model 4', 380, 341),
    ('plate 5', 'make 5', 'model 5', 205, 285),
    ('plate 6', 'make 6', 'model 6', 141, 127),
    ('plate 7', 'make 7', 'model 7', 82, 312),
    ('plate 8', 'make 8', 'model 8', 295, 65),
    ('plate 9', 'make 9', 'model 9', 189, 279),
    ('plate 10', 'make 10', 'model 10', 57, 138),
    ('plate 11', 'make 11', 'model 11', 415, 267),
    ('plate 12', 'make 12', 'model 12', 226, 76);

-- All vehicles, newest first.
SELECT * FROM vehicles ORDER BY created_at DESC LIMIT 50;

-- One vehicle by id.
SELECT * FROM vehicles WHERE id = $1;

-- How many vehicles were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM vehicles
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE drivers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    license_number VARCHAR(255) NOT NULL,
    rating NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO drivers (first_name, last_name, license_number, rating) VALUES
    ('Ben', 'Moore', 'license number 1', 452.84),
    ('Zoe', 'Smith', 'license number 2', 984.66),
    ('Jack', 'Green', 'license number 3', 398.18),
    ('Henry', 'Clark', 'license number 4', 882.71),
    ('Wendy', 'Taylor', 'license number 5', 987.24),
    ('Jack', 'Baker', 'license number 6', 538.53),
    ('Tara', 'Green', 'license number 7', 351.9),
    ('Paula', 'Wright', 'license number 8', 114.01),
    ('Rosa', 'Green', 'license number 9', 874.9),
    ('Chloe', 'Wright', 'license number 10', 697.55),
    ('Quinn', 'Wilson', 'license number 11', 280.24),
    ('Quinn', 'Baker', 'license number 12', 243.21);

-- All drivers, newest first.
SELECT * FROM drivers ORDER BY created_at DESC LIMIT 50;

-- One driver by id.
SELECT * FROM drivers WHERE id = $1;

-- How many drivers were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM drivers
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE trips (
    id SERIAL PRIMARY KEY,
    vehicle_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    distance_km NUMERIC(12, 2) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    fare NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trips_vehicle_id ON trips (vehicle_id);
CREATE INDEX idx_trips_driver_id ON trips (driver_id);

INSERT INTO trips (vehicle_id, driver_id, distance_km, started_at, fare) VALUES
    (423, 55, 251.47, '2026-03-18 01:42:00', 827.83),
    (104, 304, 171.36, '2026-06-18 00:20:00', 354.42),
    (357, 494, 833.56, '2026-05-03 15:02:00', 546.51),
    (488, 417, 710.77, '2026-11-19 18:25:00', 626.6),
    (495, 49, 826.59, '2026-04-25 01:21:00', 220.68),
    (156, 183, 565.18, '2026-08-07 20:28:00', 777.59),
    (159, 126, 492.55, '2026-02-16 10:52:00', 438.32),
    (482, 497, 829.19, '2026-12-05 19:31:00', 415.58),
    (380, 228, 634.33, '2026-04-04 05:32:00', 77.67),
    (153, 366, 578.51, '2026-01-13 17:29:00', 762.81),
    (222, 233, 65.55, '2026-12-06 12:38:00', 182.1),
    (124, 80, 699.23, '2026-06-10 09:47:00', 992.42);

-- All trips, newest first.
SELECT * FROM trips ORDER BY created_at DESC LIMIT 50;

-- One trip by id.
SELECT * FROM trips WHERE id = $1;

-- How many trips were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM trips
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    birth_date TIMESTAMP NOT NULL,
    phone VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO patients (first_name, last_name, birth_date, phone) VALUES
    ('Grace', 'Baker', '2026-10-06 01:06:00', '+49 30 3577644'),
    ('Wendy', 'Green', '2026-07-07 16:04:00', '+49 30 3004607'),
    ('Quinn', 'Parker', '2026-01-15 19:01:00', '+49 30 9020874'),
    ('Yusuf', 'Young', '2026-02-09 04:54:00', '+49 30 1901061'),
    ('Ben', 'Smith', '2026-04-02 07:04:00', '+49 30 8178249'),
    ('Yusuf', 'Adams', '2026-07-26 17:27:00', '+49 30 1060896'),
    ('Mia', 'Parker', '2026-02-05 20:01:00', '+49 30 1806250'),
    ('Liam', 'Wright', '2026-09-21 11:14:00', '+49 30 8602671'),
    ('Jack', 'Lewis', '2026-09-26 06:21:00', '+49 30 4992038'),
    ('Anna', 'Green', '2026-01-11 03:31:00', '+49 30 1063673'),
    ('Mia', 'Wilson', '2026-12-09 09:04:00', '+49 30 8530555'),
    ('Paula', 'Green', '2026-05-20 00:35:00', '+49 30 8104137');

-- All patients, newest first.
SELECT * FROM patients ORDER BY created_at DESC LIMIT 50;

-- One patient by id.
SELECT * FROM patients WHERE id = $1;

-- How many patients were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM patients
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE doctors (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    specialty VARCHAR(255) NOT NULL,
    room VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO doctors (first_name, last_name, specialty, room) VALUES
    ('Chloe', 'Mitchell', 'specialty 1', 'room 1'),
    ('Tara', 'Brown', 'specialty 2', 'room 2'),
    ('Mark', 'Hall', 'specialty 3', 'room 3'),
    ('Mark', 'Taylor', 'specialty 4', 'room 4'),
    ('Chloe', 'Baker', 'specialty 5', 'room 5'),
    ('Victor', 'Wilson', 'specialty 6', 'room 6'),
    ('Isla', 'Wright', 'specialty 7', 'room 7'),
    ('Nora', 'Carter', 'specialty 8', 'room 8'),
    ('Isla', 'Smith', 'specialty 9', 'room 9'),
    ('Emma', 'Collins', 'specialty 10', 'room 10'),
    ('Olivia', 'Evans', 'specialty 11', 'room 11'),
    ('Chloe', 'Miller', 'specialty 12', 'room 12');

-- All doctors, newest first.
SELECT * FROM doctors ORDER BY created_at DESC LIMIT 50;

-- One doctor by id.
SELECT * FROM doctors WHERE id = $1;

-- How many doctors were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM doctors
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE appointments (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    scheduled_at TIMESTAMP NOT NULL,
    reason VARCHAR(255) NOT NULL,
    confirmed BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_appointments_patient_id ON appointments (patient_id);
CREATE INDEX idx_appointments_doctor_id ON appointments (doctor_id);

INSERT INTO appointments (patient_id, doctor_id, scheduled_at, reason, confirmed) VALUES
    (135, 111, '2026-08-07 10:02:00', 'reason 1', FALSE),
    (466, 58, '2026-06-03 22:24:00', 'reason 2', TRUE),
    (229, 488, '2026-04-26 17:51:00', 'reason 3', FALSE),
    (103, 85, '2026-06-12 21:41:00', 'reason 4', FALSE),
    (190, 305, '2026-03-07 17:16:00', 'reason 5', TRUE),
    (221, 482, '2026-08-13 08:37:00', 'reason 6', FALSE),
    (494, 230, '2026-08-25 05:43:00', 'reason 7', FALSE),
    (2, 408, '2026-10-03 01:26:00', 'reason 8', TRUE),
    (37, 66, '2026-06-14 19:50:00', 'reason 9', FALSE),
    (302, 198, '2026-07-15 03:01:00', 'reason 10', FALSE),
    (376, 6, '2026-04-16 16:50:00', 'reason 11', FALSE),
    (251, 217, '2026-04-28 11:34:00', 'reason 12', TRUE);

-- All appointments, newest first.
SELECT * FROM appointments ORDER BY created_at DESC LIMIT 50;

-- One appointment by id.
SELECT * FROM appointments WHERE id = $1;

-- How many appointments were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM appointments
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    code VARCHAR(255) NOT NULL,
    credits INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_courses_code ON courses (code);
CREATE INDEX idx_courses_teacher_id ON courses (teacher_id);

INSERT INTO courses (title, code, credits, teacher_id) VALUES
    ('title 1', 'code 1', 214, 401),
    ('title 2', 'code 2', 95, 458),
    ('title 3', 'code 3', 249, 312),
    ('title 4', 'code 4', 438, 71),
    ('title 5', 'code 5', 376, 19),
    ('title 6', 'code 6', 200, 172),
    ('title 7', 'code 7', 450, 47),
    ('title 8', 'code 8', 474, 438),
    ('title 9', 'code 9', 458, 66),
    ('title 10', 'code 10', 289, 201),
    ('title 11', 'code 11', 308, 108),
    ('title 12', 'code 12', 11, 293);

-- All courses, newest first.
SELECT * FROM courses ORDER BY created_at DESC LIMIT 50;

-- One course by id.
SELECT * FROM courses WHERE id = $1;

-- How many courses were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM courses
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    enrolled_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_students_email ON students (email);

INSERT INTO students (first_name, last_name, email, enrolled_at) VALUES
    ('Chloe', 'Carter', 'paula.brown1@example.com', '2026-03-19 04:14:00'),
    ('Emma', 'Taylor', 'felix.taylor2@example.com', '2026-09-19 07:21:00'),
    ('Zoe', 'King', 'isla.parker3@example.com', '2026-05-02 19:28:00'),
    ('Rosa', 'Parker', 'lena.hall4@example.com', '2026-04-15 13:55:00'),
    ('Wendy', 'Carter', 'jack.hall5@example.com', '2026-05-10 00:51:00'),
    ('Grace', 'Taylor', 'grace.king6@example.com', '2026-09-18 08:31:00'),
    ('Anna', 'Smith', 'yusuf.evans7@example.com', '2026-03-23 15:53:00'),
    ('David', 'Carter', 'wendy.wilson8@example.com', '2026-09-10 13:21:00'),
    ('Mia', 'Adams', 'zoe.johnson9@example.com', '2026-02-16 05:35:00'),
    ('Henry', 'Mitchell', 'quinn.taylor10@example.com', '2026-12-06 17:40:00'),
    ('Wendy', 'Carter', 'emma.wright11@example.com', '2026-11-08 05:48:00'),
    ('Mark', 'Smith', 'noah.carter12@example.com', '2026-03-12 04:57:00');

-- All students, newest first.
SELECT * FROM students ORDER BY created_at DESC LIMIT 50;

-- One student by id.
SELECT * FROM students WHERE id = $1;

-- How many students were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM students
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE enrollments (
    id SERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    grade NUMERIC(12, 2) NOT NULL,
    completed BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_enrollments_student_id ON enrollments (student_id);
CREATE INDEX idx_enrollments_course_id ON enrollments (course_id);

INSERT INTO enrollments (student_id, course_id, grade, completed) VALUES
    (366, 113, 879.01, TRUE),
    (500, 465, 193.04, FALSE),
    (272, 342, 29.93, TRUE),
    (40, 484, 954.87, TRUE),
    (166, 311, 415.39, TRUE),
    (27, 192, 855.77, TRUE),
    (433, 180, 20.39, TRUE),
    (405, 483, 286.02, TRUE),
    (323, 266, 275.0, TRUE),
    (403, 315, 221.84, FALSE),
    (268, 468, 981.52, TRUE),
    (464, 66, 279.82, TRUE);

-- All enrollments, newest first.
SELECT * FROM enrollments ORDER BY created_at DESC LIMIT 50;

-- One enrollment by id.
SELECT * FROM enrollments WHERE id = $1;

-- How many enrollments were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM enrollments
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE rooms (
    id SERIAL PRIMARY KEY,
    number VARCHAR(255) NOT NULL,
    floor INTEGER NOT NULL,
    beds INTEGER NOT NULL,
    price_per_night NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO rooms (number, floor, beds, price_per_night) VALUES
    ('number 1', 346, 142, 781.61),
    ('number 2', 472, 430, 534.07),
    ('number 3', 101, 349, 138.57),
    ('number 4', 77, 326, 628.08),
    ('number 5', 243, 352, 390.98),
    ('number 6', 428, 492, 159.48),
    ('number 7', 375, 172, 584.61),
    ('number 8', 267, 298, 402.25),
    ('number 9', 402, 300, 703.73),
    ('number 10', 457, 394, 618.92),
    ('number 11', 397, 388, 156.88),
    ('number 12', 68, 341, 631.86);

-- All rooms, newest first.
SELECT * FROM rooms ORDER BY created_at DESC LIMIT 50;

-- One room by id.
SELECT * FROM rooms WHERE id = $1;

-- How many rooms were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM rooms
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE bookings (
    id SERIAL PRIMARY KEY,
    room_id INTEGER NOT NULL,
    guest_name VARCHAR(255) NOT NULL,
    check_in TIMESTAMP NOT NULL,
    check_out TIMESTAMP NOT NULL,
    paid BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_bookings_room_id ON bookings (room_id);

INSERT INTO bookings (room_id, guest_name, check_in, check_out, paid) VALUES
    (351, 'Wendy Collins', '2026-06-23 20:27:00', '2026-08-19 11:14:00', FALSE),
    (43, 'Zoe Baker', '2026-12-16 22:01:00', '2026-01-20 14:40:00', FALSE),
    (134, 'Chloe Turner', '2026-05-10 00:18:00', '2026-11-22 10:44:00', TRUE),
    (113, 'Zoe Wilson', '2026-09-08 13:51:00', '2026-07-23 08:35:00', TRUE),
    (323, 'Nora Wilson', '2026-08-19 23:06:00', '2026-11-08 21:25:00', TRUE),
    (380, 'Henry Turner', '2026-07-18 18:55:00', '2026-01-18 20:19:00', TRUE),
    (304, 'Henry Mitchell', '2026-01-24 03:18:00', '2026-04-10 02:55:00', FALSE),
    (162, 'Wendy Lewis', '2026-05-28 00:32:00', '2026-09-19 18:27:00', FALSE),
    (342, 'Lena Miller', '2026-09-20 03:08:00', '2026-07-04 12:58:00', FALSE),
    (71, 'Sam Nelson', '2026-03-28 14:28:00', '2026-04-24 15:29:00', TRUE),
    (166, 'Sam Parker', '2026-05-26 19:33:00', '2026-09-26 16:02:00', TRUE),
    (31, 'Tara Turner', '2026-03-08 12:03:00', '2026-07-13 01:24:00', FALSE);

-- All bookings, newest first.
SELECT * FROM bookings ORDER BY created_at DESC LIMIT 50;

-- One booking by id.
SELECT * FROM bookings WHERE id = $1;

-- How many bookings were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM bookings
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    plan VARCHAR(255) NOT NULL,
    price NUMERIC(12, 2) NOT NULL,
    renews_at TIMESTAMP NOT NULL,
    is_active BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_subscriptions_user_id ON subscriptions (user_id);

INSERT INTO subscriptions (user_id, plan, price, renews_at, is_active) VALUES
    (292, 'plan 1', 527.13, '2026-05-11 16:38:00', TRUE),
    (86, 'plan 2', 888.29, '2026-10-10 18:32:00', FALSE),
    (303, 'plan 3', 258.83, '2026-12-20 04:02:00', TRUE),
    (78, 'plan 4', 683.66, '2026-01-04 19:47:00', TRUE),
    (223, 'plan 5', 983.64, '2026-03-20 21:17:00', TRUE),
    (415, 'plan 6', 766.87, '2026-04-13 15:32:00', FALSE),
    (284, 'plan 7', 674.22, '2026-11-25 06:12:00', FALSE),
    (190, 'plan 8', 815.58, '2026-05-12 10:18:00', FALSE),
    (86, 'plan 9', 894.61, '2026-04-10 17:02:00', FALSE),
    (39, 'plan 10', 81.5, '2026-11-13 15:31:00', TRUE),
    (404, 'plan 11', 200.82, '2026-06-22 21:39:00', FALSE),
    (281, 'plan 12', 574.83, '2026-11-04 04:19:00', TRUE);

-- All subscriptions, newest first.
SELECT * FROM subscriptions ORDER BY created_at DESC LIMIT 50;

-- One subscription by id.
SELECT * FROM subscriptions WHERE id = $1;

-- How many subscriptions were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM subscriptions
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    message VARCHAR(255) NOT NULL,
    read BOOLEAN NOT NULL,
    sent_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notifications_user_id ON notifications (user_id);

INSERT INTO notifications (user_id, title, message, read, sent_at) VALUES
    (415, 'title 1', 'message 1', TRUE, '2026-11-02 09:43:00'),
    (127, 'title 2', 'message 2', FALSE, '2026-06-21 04:31:00'),
    (125, 'title 3', 'message 3', FALSE, '2026-12-16 08:06:00'),
    (397, 'title 4', 'message 4', FALSE, '2026-05-13 08:37:00'),
    (257, 'title 5', 'message 5', TRUE, '2026-12-14 17:28:00'),
    (489, 'title 6', 'message 6', FALSE, '2026-05-03 06:46:00'),
    (497, 'title 7', 'message 7', FALSE, '2026-02-01 10:47:00'),
    (278, 'title 8', 'message 8', TRUE, '2026-09-21 16:16:00'),
    (101, 'title 9', 'message 9', FALSE, '2026-02-11 16:25:00'),
    (461, 'title 10', 'message 10', TRUE, '2026-07-16 07:14:00'),
    (374, 'title 11', 'message 11', FALSE, '2026-09-21 08:02:00'),
    (144, 'title 12', 'message 12', TRUE, '2026-11-13 05:16:00');

-- All notifications, newest first.
SELECT * FROM notifications ORDER BY created_at DESC LIMIT 50;

-- One notification by id.
SELECT * FROM notifications WHERE id = $1;

-- How many notifications were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM notifications
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    token VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    ip_address VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user_id ON sessions (user_id);

INSERT INTO sessions (user_id, token, expires_at, ip_address) VALUES
    (27, 'token 1', '2026-12-19 15:16:00', 'ip address 1'),
    (263, 'token 2', '2026-02-07 03:48:00', 'ip address 2'),
    (329, 'token 3', '2026-03-27 23:49:00', 'ip address 3'),
    (199, 'token 4', '2026-07-16 23:44:00', 'ip address 4'),
    (130, 'token 5', '2026-01-14 18:01:00', 'ip address 5'),
    (319, 'token 6', '2026-08-17 05:22:00', 'ip address 6'),
    (13, 'token 7', '2026-07-05 23:53:00', 'ip address 7'),
    (206, 'token 8', '2026-09-21 03:50:00', 'ip address 8'),
    (460, 'token 9', '2026-12-24 12:31:00', 'ip address 9'),
    (249, 'token 10', '2026-01-02 21:04:00', 'ip address 10'),
    (253, 'token 11', '2026-03-25 00:46:00', 'ip address 11'),
    (69, 'token 12', '2026-03-27 19:05:00', 'ip address 12');

-- All sessions, newest first.
SELECT * FROM sessions ORDER BY created_at DESC LIMIT 50;

-- One session by id.
SELECT * FROM sessions WHERE id = $1;

-- How many sessions were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM sessions
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    actor_id INTEGER NOT NULL,
    action VARCHAR(255) NOT NULL,
    target VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_actor_id ON audit_logs (actor_id);

INSERT INTO audit_logs (actor_id, action, target, created_at) VALUES
    (416, 'action 1', 'target 1', '2026-12-20 11:37:00'),
    (304, 'action 2', 'target 2', '2026-09-25 11:29:00'),
    (461, 'action 3', 'target 3', '2026-09-26 00:58:00'),
    (95, 'action 4', 'target 4', '2026-06-14 07:57:00'),
    (102, 'action 5', 'target 5', '2026-02-20 09:34:00'),
    (359, 'action 6', 'target 6', '2026-09-26 20:00:00'),
    (215, 'action 7', 'target 7', '2026-10-15 02:37:00'),
    (185, 'action 8', 'target 8', '2026-01-09 11:35:00'),
    (470, 'action 9', 'target 9', '2026-09-15 08:15:00'),
    (224, 'action 10', 'target 10', '2026-05-04 08:32:00'),
    (412, 'action 11', 'target 11', '2026-01-23 16:20:00'),
    (287, 'action 12', 'target 12', '2026-02-25 16:32:00');

-- All audit logs, newest first.
SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 50;

-- One audit log by id.
SELECT * FROM audit_logs WHERE id = $1;

-- How many audit logs were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM audit_logs
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE settings (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) NOT NULL,
    value VARCHAR(255) NOT NULL,
    description VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO settings (key, value, description) VALUES
    ('key 1', 'value 1', 'description 1'),
    ('key 2', 'value 2', 'description 2'),
    ('key 3', 'value 3', 'description 3'),
    ('key 4', 'value 4', 'description 4'),
    ('key 5', 'value 5', 'description 5'),
    ('key 6', 'value 6', 'description 6'),
    ('key 7', 'value 7', 'description 7'),
    ('key 8', 'value 8', 'description 8'),
    ('key 9', 'value 9', 'description 9'),
    ('key 10', 'value 10', 'description 10'),
    ('key 11', 'value 11', 'description 11'),
    ('key 12', 'value 12', 'description 12');

-- All settings, newest first.
SELECT * FROM settings ORDER BY created_at DESC LIMIT 50;

-- One setting by id.
SELECT * FROM settings WHERE id = $1;

-- How many settings were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM settings
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE recipes (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    servings INTEGER NOT NULL,
    prep_minutes INTEGER NOT NULL,
    cook_minutes INTEGER NOT NULL,
    vegetarian BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO recipes (title, servings, prep_minutes, cook_minutes, vegetarian) VALUES
    ('title 1', 24, 302, 28, FALSE),
    ('title 2', 170, 85, 81, TRUE),
    ('title 3', 118, 348, 41, FALSE),
    ('title 4', 52, 383, 496, FALSE),
    ('title 5', 175, 448, 362, FALSE),
    ('title 6', 449, 375, 46, TRUE),
    ('title 7', 19, 357, 212, FALSE),
    ('title 8', 180, 7, 335, TRUE),
    ('title 9', 405, 187, 95, FALSE),
    ('title 10', 336, 494, 79, TRUE),
    ('title 11', 354, 112, 361, TRUE),
    ('title 12', 282, 69, 469, TRUE);

-- All recipes, newest first.
SELECT * FROM recipes ORDER BY created_at DESC LIMIT 50;

-- One recipe by id.
SELECT * FROM recipes WHERE id = $1;

-- How many recipes were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM recipes
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE ingredients (
    id SERIAL PRIMARY KEY,
    recipe_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL,
    unit VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ingredients_recipe_id ON ingredients (recipe_id);

INSERT INTO ingredients (recipe_id, name, amount, unit) VALUES
    (165, 'Grace Young', 461.44, 'unit 1'),
    (271, 'Wendy Clark', 475.16, 'unit 2'),
    (101, 'Isla Walker', 814.57, 'unit 3'),
    (477, 'Chloe Parker', 928.62, 'unit 4'),
    (200, 'Nora Hall', 415.9, 'unit 5'),
    (340, 'Felix Lewis', 681.17, 'unit 6'),
    (250, 'Victor King', 761.94, 'unit 7'),
    (396, 'Felix Moore', 742.86, 'unit 8'),
    (323, 'Isla Smith', 527.79, 'unit 9'),
    (313, 'Wendy Collins', 404.64, 'unit 10'),
    (20, 'Anna Lewis', 919.08, 'unit 11'),
    (483, 'Grace Lewis', 383.08, 'unit 12');

-- All ingredients, newest first.
SELECT * FROM ingredients ORDER BY created_at DESC LIMIT 50;

-- One ingredient by id.
SELECT * FROM ingredients WHERE id = $1;

-- How many ingredients were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM ingredients
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE playlists (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    owner_id INTEGER NOT NULL,
    is_public BOOLEAN NOT NULL,
    track_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_playlists_owner_id ON playlists (owner_id);

INSERT INTO playlists (name, owner_id, is_public, track_count) VALUES
    ('Anna Collins', 210, FALSE, 209),
    ('Victor Hall', 187, TRUE, 42),
    ('Liam Johnson', 238, FALSE, 286),
    ('Quinn Evans', 145, FALSE, 51),
    ('Henry King', 401, FALSE, 153),
    ('Noah Taylor', 450, TRUE, 278),
    ('Tara Wright', 124, TRUE, 358),
    ('Liam Mitchell', 27, TRUE, 393),
    ('Lucas Wright', 426, TRUE, 18),
    ('Mia Young', 222, FALSE, 282),
    ('Noah Young', 10, FALSE, 177),
    ('Lucas Taylor', 38, TRUE, 26);

-- All playlists, newest first.
SELECT * FROM playlists ORDER BY created_at DESC LIMIT 50;

-- One playlist by id.
SELECT * FROM playlists WHERE id = $1;

-- How many playlists were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM playlists
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE tracks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist VARCHAR(255) NOT NULL,
    album VARCHAR(255) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO tracks (title, artist, album, duration_seconds) VALUES
    ('title 1', 'artist 1', 'album 1', 436),
    ('title 2', 'artist 2', 'album 2', 380),
    ('title 3', 'artist 3', 'album 3', 236),
    ('title 4', 'artist 4', 'album 4', 327),
    ('title 5', 'artist 5', 'album 5', 462),
    ('title 6', 'artist 6', 'album 6', 302),
    ('title 7', 'artist 7', 'album 7', 477),
    ('title 8', 'artist 8', 'album 8', 6),
    ('title 9', 'artist 9', 'album 9', 25),
    ('title 10', 'artist 10', 'album 10', 151),
    ('title 11', 'artist 11', 'album 11', 395),
    ('title 12', 'artist 12', 'album 12', 114);

-- All tracks, newest first.
SELECT * FROM tracks ORDER BY created_at DESC LIMIT 50;

-- One track by id.
SELECT * FROM tracks WHERE id = $1;

-- How many tracks were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM tracks
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

CREATE TABLE devices (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    serial_number VARCHAR(255) NOT NULL,
    firmware VARCHAR(255) NOT NULL,
    online BOOLEAN NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO devices (name, serial_number, firmware, online, last_seen) VALUES
    ('Tara Green', 'serial number 1', 'firmware 1', TRUE, '2026-02-21 20:33:00'),
    ('Oscar Baker', 'serial number 2', 'firmware 2', FALSE, '2026-03-09 22:10:00'),
    ('Mark Miller', 'serial number 3', 'firmware 3', FALSE, '2026-12-01 10:37:00'),
    ('Yusuf Nelson', 'serial number 4', 'firmware 4', FALSE, '2026-02-17 01:47:00'),
    ('Anna Carter', 'serial number 5', 'firmware 5', FALSE, '2026-02-18 21:03:00'),
    ('Rosa Wright', 'serial number 6', 'firmware 6', TRUE, '2026-06-24 08:21:00'),
    ('Nora Walker', 'serial number 7', 'firmware 7', TRUE, '2026-03-19 07:35:00'),
    ('Isla King', 'serial number 8', 'firmware 8', FALSE, '2026-03-19 21:51:00'),
    ('Emma Brown', 'serial number 9', 'firmware 9', TRUE, '2026-05-02 17:37:00'),
    ('Felix Taylor', 'serial number 10', 'firmware 10', TRUE, '2026-03-17 04:31:00'),
    ('Chloe Turner', 'serial number 11', 'firmware 11', FALSE, '2026-05-07 08:12:00'),
    ('Wendy Smith', 'serial number 12', 'firmware 12', FALSE, '2026-09-20 01:56:00');

-- All devices, newest first.
SELECT * FROM devices ORDER BY created_at DESC LIMIT 50;

-- One device by id.
SELECT * FROM devices WHERE id = $1;

-- How many devices were created per day in the last 30 days.
SELECT DATE(created_at) AS day, COUNT(*) AS created
FROM devices
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY day;

COMMIT;
