-- Sample E-Commerce Schema (used in fixture tests)

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    first_name    TEXT    NOT NULL,
    last_name     TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    country       TEXT    NOT NULL DEFAULT 'US',
    created_at    TEXT    NOT NULL
);

CREATE TABLE categories (
    category_id  INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL UNIQUE,
    description  TEXT
);

CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    category_id  INTEGER NOT NULL REFERENCES categories(category_id),
    price        REAL    NOT NULL,
    stock_qty    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    status       TEXT    NOT NULL DEFAULT 'pending',
    created_at   TEXT    NOT NULL,
    total_amount REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE order_items (
    item_id     INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(order_id),
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    quantity    INTEGER NOT NULL,
    unit_price  REAL    NOT NULL
);
