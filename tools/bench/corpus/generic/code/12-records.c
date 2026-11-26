/* Plain C records and growable arrays, one per record type. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NAME_LEN 64

/* A user. */
typedef struct {
    long id;
    char username[NAME_LEN];
    char email[NAME_LEN];
    char full_name[NAME_LEN];
    bool is_active;
    time_t created_at;
} user_t;

typedef struct {
    user_t *items;
    size_t count;
    size_t capacity;
} user_list_t;

/* Initialise an empty list. */
void user_list_init(user_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int user_list_push(user_list_t *list, const user_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        user_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the user with this id, or NULL. */
user_t *user_list_find(user_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one user on one line. */
void user_print(const user_t *item) {
    printf("User(id=%ld, username=%s, email=%s, full_name=%s, is_active=%s, created_at=%ld)\n", item->id, item->username, item->email, item->full_name, item->is_active ? "true" : "false", (long)item->created_at);
}

/* Release the memory of the list. */
void user_list_free(user_list_t *list) {
    free(list->items);
    user_list_init(list);
}

/* A product. */
typedef struct {
    long id;
    char sku[NAME_LEN];
    char name[NAME_LEN];
    char description[NAME_LEN];
    double price;
    int stock;
} product_t;

typedef struct {
    product_t *items;
    size_t count;
    size_t capacity;
} product_list_t;

/* Initialise an empty list. */
void product_list_init(product_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int product_list_push(product_list_t *list, const product_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        product_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the product with this id, or NULL. */
product_t *product_list_find(product_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one product on one line. */
void product_print(const product_t *item) {
    printf("Product(id=%ld, sku=%s, name=%s, description=%s, price=%.2f, stock=%d)\n", item->id, item->sku, item->name, item->description, item->price, item->stock);
}

/* Release the memory of the list. */
void product_list_free(product_list_t *list) {
    free(list->items);
    product_list_init(list);
}

/* A order. */
typedef struct {
    long id;
    int customer_id;
    char status[NAME_LEN];
    double total;
    char currency[NAME_LEN];
    time_t placed_at;
} order_t;

typedef struct {
    order_t *items;
    size_t count;
    size_t capacity;
} order_list_t;

/* Initialise an empty list. */
void order_list_init(order_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int order_list_push(order_list_t *list, const order_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        order_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the order with this id, or NULL. */
order_t *order_list_find(order_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one order on one line. */
void order_print(const order_t *item) {
    printf("Order(id=%ld, customer_id=%d, status=%s, total=%.2f, currency=%s, placed_at=%ld)\n", item->id, item->customer_id, item->status, item->total, item->currency, (long)item->placed_at);
}

/* Release the memory of the list. */
void order_list_free(order_list_t *list) {
    free(list->items);
    order_list_init(list);
}

/* A order item. */
typedef struct {
    long id;
    int order_id;
    int product_id;
    int quantity;
    double unit_price;
} order_item_t;

typedef struct {
    order_item_t *items;
    size_t count;
    size_t capacity;
} order_item_list_t;

/* Initialise an empty list. */
void order_item_list_init(order_item_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int order_item_list_push(order_item_list_t *list, const order_item_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        order_item_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the order item with this id, or NULL. */
order_item_t *order_item_list_find(order_item_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one order item on one line. */
void order_item_print(const order_item_t *item) {
    printf("OrderItem(id=%ld, order_id=%d, product_id=%d, quantity=%d, unit_price=%.2f)\n", item->id, item->order_id, item->product_id, item->quantity, item->unit_price);
}

/* Release the memory of the list. */
void order_item_list_free(order_item_list_t *list) {
    free(list->items);
    order_item_list_init(list);
}

/* A customer. */
typedef struct {
    long id;
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    char email[NAME_LEN];
    char phone[NAME_LEN];
    char country[NAME_LEN];
} customer_t;

typedef struct {
    customer_t *items;
    size_t count;
    size_t capacity;
} customer_list_t;

/* Initialise an empty list. */
void customer_list_init(customer_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int customer_list_push(customer_list_t *list, const customer_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        customer_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the customer with this id, or NULL. */
customer_t *customer_list_find(customer_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one customer on one line. */
void customer_print(const customer_t *item) {
    printf("Customer(id=%ld, first_name=%s, last_name=%s, email=%s, phone=%s, country=%s)\n", item->id, item->first_name, item->last_name, item->email, item->phone, item->country);
}

/* Release the memory of the list. */
void customer_list_free(customer_list_t *list) {
    free(list->items);
    customer_list_init(list);
}

/* A invoice. */
typedef struct {
    long id;
    int order_id;
    char number[NAME_LEN];
    double amount;
    time_t due_date;
    bool paid;
} invoice_t;

typedef struct {
    invoice_t *items;
    size_t count;
    size_t capacity;
} invoice_list_t;

/* Initialise an empty list. */
void invoice_list_init(invoice_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int invoice_list_push(invoice_list_t *list, const invoice_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        invoice_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the invoice with this id, or NULL. */
invoice_t *invoice_list_find(invoice_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one invoice on one line. */
void invoice_print(const invoice_t *item) {
    printf("Invoice(id=%ld, order_id=%d, number=%s, amount=%.2f, due_date=%ld, paid=%s)\n", item->id, item->order_id, item->number, item->amount, (long)item->due_date, item->paid ? "true" : "false");
}

/* Release the memory of the list. */
void invoice_list_free(invoice_list_t *list) {
    free(list->items);
    invoice_list_init(list);
}

/* A payment. */
typedef struct {
    long id;
    int invoice_id;
    char method[NAME_LEN];
    double amount;
    char reference[NAME_LEN];
    time_t received_at;
} payment_t;

typedef struct {
    payment_t *items;
    size_t count;
    size_t capacity;
} payment_list_t;

/* Initialise an empty list. */
void payment_list_init(payment_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int payment_list_push(payment_list_t *list, const payment_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        payment_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the payment with this id, or NULL. */
payment_t *payment_list_find(payment_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one payment on one line. */
void payment_print(const payment_t *item) {
    printf("Payment(id=%ld, invoice_id=%d, method=%s, amount=%.2f, reference=%s, received_at=%ld)\n", item->id, item->invoice_id, item->method, item->amount, item->reference, (long)item->received_at);
}

/* Release the memory of the list. */
void payment_list_free(payment_list_t *list) {
    free(list->items);
    payment_list_init(list);
}

/* A category. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char slug[NAME_LEN];
    int parent_id;
    int position;
} category_t;

typedef struct {
    category_t *items;
    size_t count;
    size_t capacity;
} category_list_t;

/* Initialise an empty list. */
void category_list_init(category_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int category_list_push(category_list_t *list, const category_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        category_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the category with this id, or NULL. */
category_t *category_list_find(category_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one category on one line. */
void category_print(const category_t *item) {
    printf("Category(id=%ld, name=%s, slug=%s, parent_id=%d, position=%d)\n", item->id, item->name, item->slug, item->parent_id, item->position);
}

/* Release the memory of the list. */
void category_list_free(category_list_t *list) {
    free(list->items);
    category_list_init(list);
}

/* A review. */
typedef struct {
    long id;
    int product_id;
    int user_id;
    int rating;
    char title[NAME_LEN];
    char body[NAME_LEN];
} review_t;

typedef struct {
    review_t *items;
    size_t count;
    size_t capacity;
} review_list_t;

/* Initialise an empty list. */
void review_list_init(review_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int review_list_push(review_list_t *list, const review_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        review_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the review with this id, or NULL. */
review_t *review_list_find(review_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one review on one line. */
void review_print(const review_t *item) {
    printf("Review(id=%ld, product_id=%d, user_id=%d, rating=%d, title=%s, body=%s)\n", item->id, item->product_id, item->user_id, item->rating, item->title, item->body);
}

/* Release the memory of the list. */
void review_list_free(review_list_t *list) {
    free(list->items);
    review_list_init(list);
}

/* A address. */
typedef struct {
    long id;
    int customer_id;
    char street[NAME_LEN];
    char city[NAME_LEN];
    char postal_code[NAME_LEN];
    char country[NAME_LEN];
} address_t;

typedef struct {
    address_t *items;
    size_t count;
    size_t capacity;
} address_list_t;

/* Initialise an empty list. */
void address_list_init(address_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int address_list_push(address_list_t *list, const address_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        address_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the address with this id, or NULL. */
address_t *address_list_find(address_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one address on one line. */
void address_print(const address_t *item) {
    printf("Address(id=%ld, customer_id=%d, street=%s, city=%s, postal_code=%s, country=%s)\n", item->id, item->customer_id, item->street, item->city, item->postal_code, item->country);
}

/* Release the memory of the list. */
void address_list_free(address_list_t *list) {
    free(list->items);
    address_list_init(list);
}

/* A shipment. */
typedef struct {
    long id;
    int order_id;
    char carrier[NAME_LEN];
    char tracking_number[NAME_LEN];
    time_t shipped_at;
    bool delivered;
} shipment_t;

typedef struct {
    shipment_t *items;
    size_t count;
    size_t capacity;
} shipment_list_t;

/* Initialise an empty list. */
void shipment_list_init(shipment_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int shipment_list_push(shipment_list_t *list, const shipment_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        shipment_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the shipment with this id, or NULL. */
shipment_t *shipment_list_find(shipment_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one shipment on one line. */
void shipment_print(const shipment_t *item) {
    printf("Shipment(id=%ld, order_id=%d, carrier=%s, tracking_number=%s, shipped_at=%ld, delivered=%s)\n", item->id, item->order_id, item->carrier, item->tracking_number, (long)item->shipped_at, item->delivered ? "true" : "false");
}

/* Release the memory of the list. */
void shipment_list_free(shipment_list_t *list) {
    free(list->items);
    shipment_list_init(list);
}

/* A warehouse. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char city[NAME_LEN];
    int capacity;
    bool is_open;
} warehouse_t;

typedef struct {
    warehouse_t *items;
    size_t count;
    size_t capacity;
} warehouse_list_t;

/* Initialise an empty list. */
void warehouse_list_init(warehouse_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int warehouse_list_push(warehouse_list_t *list, const warehouse_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        warehouse_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the warehouse with this id, or NULL. */
warehouse_t *warehouse_list_find(warehouse_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one warehouse on one line. */
void warehouse_print(const warehouse_t *item) {
    printf("Warehouse(id=%ld, name=%s, city=%s, capacity=%d, is_open=%s)\n", item->id, item->name, item->city, item->capacity, item->is_open ? "true" : "false");
}

/* Release the memory of the list. */
void warehouse_list_free(warehouse_list_t *list) {
    free(list->items);
    warehouse_list_init(list);
}

/* A supplier. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char contact_email[NAME_LEN];
    char phone[NAME_LEN];
    char country[NAME_LEN];
    int rating;
} supplier_t;

typedef struct {
    supplier_t *items;
    size_t count;
    size_t capacity;
} supplier_list_t;

/* Initialise an empty list. */
void supplier_list_init(supplier_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int supplier_list_push(supplier_list_t *list, const supplier_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        supplier_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the supplier with this id, or NULL. */
supplier_t *supplier_list_find(supplier_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one supplier on one line. */
void supplier_print(const supplier_t *item) {
    printf("Supplier(id=%ld, name=%s, contact_email=%s, phone=%s, country=%s, rating=%d)\n", item->id, item->name, item->contact_email, item->phone, item->country, item->rating);
}

/* Release the memory of the list. */
void supplier_list_free(supplier_list_t *list) {
    free(list->items);
    supplier_list_init(list);
}

/* A coupon. */
typedef struct {
    long id;
    char code[NAME_LEN];
    int discount_percent;
    time_t valid_until;
    int max_uses;
    bool is_active;
} coupon_t;

typedef struct {
    coupon_t *items;
    size_t count;
    size_t capacity;
} coupon_list_t;

/* Initialise an empty list. */
void coupon_list_init(coupon_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int coupon_list_push(coupon_list_t *list, const coupon_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        coupon_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the coupon with this id, or NULL. */
coupon_t *coupon_list_find(coupon_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one coupon on one line. */
void coupon_print(const coupon_t *item) {
    printf("Coupon(id=%ld, code=%s, discount_percent=%d, valid_until=%ld, max_uses=%d, is_active=%s)\n", item->id, item->code, item->discount_percent, (long)item->valid_until, item->max_uses, item->is_active ? "true" : "false");
}

/* Release the memory of the list. */
void coupon_list_free(coupon_list_t *list) {
    free(list->items);
    coupon_list_init(list);
}

/* A employee. */
typedef struct {
    long id;
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    char email[NAME_LEN];
    char department[NAME_LEN];
    double salary;
} employee_t;

typedef struct {
    employee_t *items;
    size_t count;
    size_t capacity;
} employee_list_t;

/* Initialise an empty list. */
void employee_list_init(employee_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int employee_list_push(employee_list_t *list, const employee_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        employee_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the employee with this id, or NULL. */
employee_t *employee_list_find(employee_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one employee on one line. */
void employee_print(const employee_t *item) {
    printf("Employee(id=%ld, first_name=%s, last_name=%s, email=%s, department=%s, salary=%.2f)\n", item->id, item->first_name, item->last_name, item->email, item->department, item->salary);
}

/* Release the memory of the list. */
void employee_list_free(employee_list_t *list) {
    free(list->items);
    employee_list_init(list);
}

/* A department. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char code[NAME_LEN];
    int manager_id;
    double budget;
} department_t;

typedef struct {
    department_t *items;
    size_t count;
    size_t capacity;
} department_list_t;

/* Initialise an empty list. */
void department_list_init(department_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int department_list_push(department_list_t *list, const department_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        department_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the department with this id, or NULL. */
department_t *department_list_find(department_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one department on one line. */
void department_print(const department_t *item) {
    printf("Department(id=%ld, name=%s, code=%s, manager_id=%d, budget=%.2f)\n", item->id, item->name, item->code, item->manager_id, item->budget);
}

/* Release the memory of the list. */
void department_list_free(department_list_t *list) {
    free(list->items);
    department_list_init(list);
}

/* A project. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char description[NAME_LEN];
    time_t start_date;
    time_t end_date;
    double budget;
} project_t;

typedef struct {
    project_t *items;
    size_t count;
    size_t capacity;
} project_list_t;

/* Initialise an empty list. */
void project_list_init(project_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int project_list_push(project_list_t *list, const project_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        project_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the project with this id, or NULL. */
project_t *project_list_find(project_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one project on one line. */
void project_print(const project_t *item) {
    printf("Project(id=%ld, name=%s, description=%s, start_date=%ld, end_date=%ld, budget=%.2f)\n", item->id, item->name, item->description, (long)item->start_date, (long)item->end_date, item->budget);
}

/* Release the memory of the list. */
void project_list_free(project_list_t *list) {
    free(list->items);
    project_list_init(list);
}

/* A task. */
typedef struct {
    long id;
    int project_id;
    char title[NAME_LEN];
    char status[NAME_LEN];
    int priority;
    time_t due_date;
} task_t;

typedef struct {
    task_t *items;
    size_t count;
    size_t capacity;
} task_list_t;

/* Initialise an empty list. */
void task_list_init(task_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int task_list_push(task_list_t *list, const task_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        task_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the task with this id, or NULL. */
task_t *task_list_find(task_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one task on one line. */
void task_print(const task_t *item) {
    printf("Task(id=%ld, project_id=%d, title=%s, status=%s, priority=%d, due_date=%ld)\n", item->id, item->project_id, item->title, item->status, item->priority, (long)item->due_date);
}

/* Release the memory of the list. */
void task_list_free(task_list_t *list) {
    free(list->items);
    task_list_init(list);
}

/* A comment. */
typedef struct {
    long id;
    int task_id;
    int author_id;
    char body[NAME_LEN];
    time_t created_at;
} comment_t;

typedef struct {
    comment_t *items;
    size_t count;
    size_t capacity;
} comment_list_t;

/* Initialise an empty list. */
void comment_list_init(comment_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int comment_list_push(comment_list_t *list, const comment_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        comment_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the comment with this id, or NULL. */
comment_t *comment_list_find(comment_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one comment on one line. */
void comment_print(const comment_t *item) {
    printf("Comment(id=%ld, task_id=%d, author_id=%d, body=%s, created_at=%ld)\n", item->id, item->task_id, item->author_id, item->body, (long)item->created_at);
}

/* Release the memory of the list. */
void comment_list_free(comment_list_t *list) {
    free(list->items);
    comment_list_init(list);
}

/* A tag. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char color[NAME_LEN];
    int usage_count;
} tag_t;

typedef struct {
    tag_t *items;
    size_t count;
    size_t capacity;
} tag_list_t;

/* Initialise an empty list. */
void tag_list_init(tag_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int tag_list_push(tag_list_t *list, const tag_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        tag_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the tag with this id, or NULL. */
tag_t *tag_list_find(tag_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one tag on one line. */
void tag_print(const tag_t *item) {
    printf("Tag(id=%ld, name=%s, color=%s, usage_count=%d)\n", item->id, item->name, item->color, item->usage_count);
}

/* Release the memory of the list. */
void tag_list_free(tag_list_t *list) {
    free(list->items);
    tag_list_init(list);
}

/* A article. */
typedef struct {
    long id;
    char title[NAME_LEN];
    char slug[NAME_LEN];
    char body[NAME_LEN];
    int author_id;
    bool published;
} article_t;

typedef struct {
    article_t *items;
    size_t count;
    size_t capacity;
} article_list_t;

/* Initialise an empty list. */
void article_list_init(article_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int article_list_push(article_list_t *list, const article_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        article_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the article with this id, or NULL. */
article_t *article_list_find(article_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one article on one line. */
void article_print(const article_t *item) {
    printf("Article(id=%ld, title=%s, slug=%s, body=%s, author_id=%d, published=%s)\n", item->id, item->title, item->slug, item->body, item->author_id, item->published ? "true" : "false");
}

/* Release the memory of the list. */
void article_list_free(article_list_t *list) {
    free(list->items);
    article_list_init(list);
}

/* A author. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char bio[NAME_LEN];
    char email[NAME_LEN];
    char website[NAME_LEN];
} author_t;

typedef struct {
    author_t *items;
    size_t count;
    size_t capacity;
} author_list_t;

/* Initialise an empty list. */
void author_list_init(author_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int author_list_push(author_list_t *list, const author_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        author_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the author with this id, or NULL. */
author_t *author_list_find(author_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one author on one line. */
void author_print(const author_t *item) {
    printf("Author(id=%ld, name=%s, bio=%s, email=%s, website=%s)\n", item->id, item->name, item->bio, item->email, item->website);
}

/* Release the memory of the list. */
void author_list_free(author_list_t *list) {
    free(list->items);
    author_list_init(list);
}

/* A book. */
typedef struct {
    long id;
    char title[NAME_LEN];
    char isbn[NAME_LEN];
    int author_id;
    int year;
    int pages;
} book_t;

typedef struct {
    book_t *items;
    size_t count;
    size_t capacity;
} book_list_t;

/* Initialise an empty list. */
void book_list_init(book_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int book_list_push(book_list_t *list, const book_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        book_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the book with this id, or NULL. */
book_t *book_list_find(book_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one book on one line. */
void book_print(const book_t *item) {
    printf("Book(id=%ld, title=%s, isbn=%s, author_id=%d, year=%d, pages=%d)\n", item->id, item->title, item->isbn, item->author_id, item->year, item->pages);
}

/* Release the memory of the list. */
void book_list_free(book_list_t *list) {
    free(list->items);
    book_list_init(list);
}

/* A library. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char city[NAME_LEN];
    int opening_hour;
    int closing_hour;
} library_t;

typedef struct {
    library_t *items;
    size_t count;
    size_t capacity;
} library_list_t;

/* Initialise an empty list. */
void library_list_init(library_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int library_list_push(library_list_t *list, const library_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        library_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the library with this id, or NULL. */
library_t *library_list_find(library_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one library on one line. */
void library_print(const library_t *item) {
    printf("Library(id=%ld, name=%s, city=%s, opening_hour=%d, closing_hour=%d)\n", item->id, item->name, item->city, item->opening_hour, item->closing_hour);
}

/* Release the memory of the list. */
void library_list_free(library_list_t *list) {
    free(list->items);
    library_list_init(list);
}

/* A loan. */
typedef struct {
    long id;
    int book_id;
    int member_id;
    time_t borrowed_at;
    time_t due_date;
    bool returned;
} loan_t;

typedef struct {
    loan_t *items;
    size_t count;
    size_t capacity;
} loan_list_t;

/* Initialise an empty list. */
void loan_list_init(loan_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int loan_list_push(loan_list_t *list, const loan_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        loan_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the loan with this id, or NULL. */
loan_t *loan_list_find(loan_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one loan on one line. */
void loan_print(const loan_t *item) {
    printf("Loan(id=%ld, book_id=%d, member_id=%d, borrowed_at=%ld, due_date=%ld, returned=%s)\n", item->id, item->book_id, item->member_id, (long)item->borrowed_at, (long)item->due_date, item->returned ? "true" : "false");
}

/* Release the memory of the list. */
void loan_list_free(loan_list_t *list) {
    free(list->items);
    loan_list_init(list);
}

/* A member. */
typedef struct {
    long id;
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    char email[NAME_LEN];
    time_t joined_at;
} member_t;

typedef struct {
    member_t *items;
    size_t count;
    size_t capacity;
} member_list_t;

/* Initialise an empty list. */
void member_list_init(member_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int member_list_push(member_list_t *list, const member_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        member_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the member with this id, or NULL. */
member_t *member_list_find(member_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one member on one line. */
void member_print(const member_t *item) {
    printf("Member(id=%ld, first_name=%s, last_name=%s, email=%s, joined_at=%ld)\n", item->id, item->first_name, item->last_name, item->email, (long)item->joined_at);
}

/* Release the memory of the list. */
void member_list_free(member_list_t *list) {
    free(list->items);
    member_list_init(list);
}

/* A event. */
typedef struct {
    long id;
    char title[NAME_LEN];
    char location[NAME_LEN];
    time_t starts_at;
    time_t ends_at;
    int capacity;
} event_t;

typedef struct {
    event_t *items;
    size_t count;
    size_t capacity;
} event_list_t;

/* Initialise an empty list. */
void event_list_init(event_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int event_list_push(event_list_t *list, const event_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        event_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the event with this id, or NULL. */
event_t *event_list_find(event_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one event on one line. */
void event_print(const event_t *item) {
    printf("Event(id=%ld, title=%s, location=%s, starts_at=%ld, ends_at=%ld, capacity=%d)\n", item->id, item->title, item->location, (long)item->starts_at, (long)item->ends_at, item->capacity);
}

/* Release the memory of the list. */
void event_list_free(event_list_t *list) {
    free(list->items);
    event_list_init(list);
}

/* A ticket. */
typedef struct {
    long id;
    int event_id;
    char holder_name[NAME_LEN];
    char seat[NAME_LEN];
    double price;
    bool checked_in;
} ticket_t;

typedef struct {
    ticket_t *items;
    size_t count;
    size_t capacity;
} ticket_list_t;

/* Initialise an empty list. */
void ticket_list_init(ticket_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int ticket_list_push(ticket_list_t *list, const ticket_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        ticket_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the ticket with this id, or NULL. */
ticket_t *ticket_list_find(ticket_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one ticket on one line. */
void ticket_print(const ticket_t *item) {
    printf("Ticket(id=%ld, event_id=%d, holder_name=%s, seat=%s, price=%.2f, checked_in=%s)\n", item->id, item->event_id, item->holder_name, item->seat, item->price, item->checked_in ? "true" : "false");
}

/* Release the memory of the list. */
void ticket_list_free(ticket_list_t *list) {
    free(list->items);
    ticket_list_init(list);
}

/* A venue. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char address[NAME_LEN];
    char city[NAME_LEN];
    int capacity;
} venue_t;

typedef struct {
    venue_t *items;
    size_t count;
    size_t capacity;
} venue_list_t;

/* Initialise an empty list. */
void venue_list_init(venue_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int venue_list_push(venue_list_t *list, const venue_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        venue_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the venue with this id, or NULL. */
venue_t *venue_list_find(venue_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one venue on one line. */
void venue_print(const venue_t *item) {
    printf("Venue(id=%ld, name=%s, address=%s, city=%s, capacity=%d)\n", item->id, item->name, item->address, item->city, item->capacity);
}

/* Release the memory of the list. */
void venue_list_free(venue_list_t *list) {
    free(list->items);
    venue_list_init(list);
}

/* A vehicle. */
typedef struct {
    long id;
    char plate[NAME_LEN];
    char make[NAME_LEN];
    char model[NAME_LEN];
    int year;
    int mileage;
} vehicle_t;

typedef struct {
    vehicle_t *items;
    size_t count;
    size_t capacity;
} vehicle_list_t;

/* Initialise an empty list. */
void vehicle_list_init(vehicle_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int vehicle_list_push(vehicle_list_t *list, const vehicle_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        vehicle_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the vehicle with this id, or NULL. */
vehicle_t *vehicle_list_find(vehicle_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one vehicle on one line. */
void vehicle_print(const vehicle_t *item) {
    printf("Vehicle(id=%ld, plate=%s, make=%s, model=%s, year=%d, mileage=%d)\n", item->id, item->plate, item->make, item->model, item->year, item->mileage);
}

/* Release the memory of the list. */
void vehicle_list_free(vehicle_list_t *list) {
    free(list->items);
    vehicle_list_init(list);
}

/* A driver. */
typedef struct {
    long id;
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    char license_number[NAME_LEN];
    double rating;
} driver_t;

typedef struct {
    driver_t *items;
    size_t count;
    size_t capacity;
} driver_list_t;

/* Initialise an empty list. */
void driver_list_init(driver_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int driver_list_push(driver_list_t *list, const driver_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        driver_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the driver with this id, or NULL. */
driver_t *driver_list_find(driver_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one driver on one line. */
void driver_print(const driver_t *item) {
    printf("Driver(id=%ld, first_name=%s, last_name=%s, license_number=%s, rating=%.2f)\n", item->id, item->first_name, item->last_name, item->license_number, item->rating);
}

/* Release the memory of the list. */
void driver_list_free(driver_list_t *list) {
    free(list->items);
    driver_list_init(list);
}

/* A trip. */
typedef struct {
    long id;
    int vehicle_id;
    int driver_id;
    double distance_km;
    time_t started_at;
    double fare;
} trip_t;

typedef struct {
    trip_t *items;
    size_t count;
    size_t capacity;
} trip_list_t;

/* Initialise an empty list. */
void trip_list_init(trip_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int trip_list_push(trip_list_t *list, const trip_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        trip_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the trip with this id, or NULL. */
trip_t *trip_list_find(trip_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one trip on one line. */
void trip_print(const trip_t *item) {
    printf("Trip(id=%ld, vehicle_id=%d, driver_id=%d, distance_km=%.2f, started_at=%ld, fare=%.2f)\n", item->id, item->vehicle_id, item->driver_id, item->distance_km, (long)item->started_at, item->fare);
}

/* Release the memory of the list. */
void trip_list_free(trip_list_t *list) {
    free(list->items);
    trip_list_init(list);
}

/* A patient. */
typedef struct {
    long id;
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    time_t birth_date;
    char phone[NAME_LEN];
} patient_t;

typedef struct {
    patient_t *items;
    size_t count;
    size_t capacity;
} patient_list_t;

/* Initialise an empty list. */
void patient_list_init(patient_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int patient_list_push(patient_list_t *list, const patient_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        patient_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the patient with this id, or NULL. */
patient_t *patient_list_find(patient_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one patient on one line. */
void patient_print(const patient_t *item) {
    printf("Patient(id=%ld, first_name=%s, last_name=%s, birth_date=%ld, phone=%s)\n", item->id, item->first_name, item->last_name, (long)item->birth_date, item->phone);
}

/* Release the memory of the list. */
void patient_list_free(patient_list_t *list) {
    free(list->items);
    patient_list_init(list);
}

/* A doctor. */
typedef struct {
    long id;
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    char specialty[NAME_LEN];
    char room[NAME_LEN];
} doctor_t;

typedef struct {
    doctor_t *items;
    size_t count;
    size_t capacity;
} doctor_list_t;

/* Initialise an empty list. */
void doctor_list_init(doctor_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int doctor_list_push(doctor_list_t *list, const doctor_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        doctor_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the doctor with this id, or NULL. */
doctor_t *doctor_list_find(doctor_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one doctor on one line. */
void doctor_print(const doctor_t *item) {
    printf("Doctor(id=%ld, first_name=%s, last_name=%s, specialty=%s, room=%s)\n", item->id, item->first_name, item->last_name, item->specialty, item->room);
}

/* Release the memory of the list. */
void doctor_list_free(doctor_list_t *list) {
    free(list->items);
    doctor_list_init(list);
}

/* A appointment. */
typedef struct {
    long id;
    int patient_id;
    int doctor_id;
    time_t scheduled_at;
    char reason[NAME_LEN];
    bool confirmed;
} appointment_t;

typedef struct {
    appointment_t *items;
    size_t count;
    size_t capacity;
} appointment_list_t;

/* Initialise an empty list. */
void appointment_list_init(appointment_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int appointment_list_push(appointment_list_t *list, const appointment_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        appointment_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the appointment with this id, or NULL. */
appointment_t *appointment_list_find(appointment_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one appointment on one line. */
void appointment_print(const appointment_t *item) {
    printf("Appointment(id=%ld, patient_id=%d, doctor_id=%d, scheduled_at=%ld, reason=%s, confirmed=%s)\n", item->id, item->patient_id, item->doctor_id, (long)item->scheduled_at, item->reason, item->confirmed ? "true" : "false");
}

/* Release the memory of the list. */
void appointment_list_free(appointment_list_t *list) {
    free(list->items);
    appointment_list_init(list);
}

/* A course. */
typedef struct {
    long id;
    char title[NAME_LEN];
    char code[NAME_LEN];
    int credits;
    int teacher_id;
} course_t;

typedef struct {
    course_t *items;
    size_t count;
    size_t capacity;
} course_list_t;

/* Initialise an empty list. */
void course_list_init(course_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int course_list_push(course_list_t *list, const course_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        course_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the course with this id, or NULL. */
course_t *course_list_find(course_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one course on one line. */
void course_print(const course_t *item) {
    printf("Course(id=%ld, title=%s, code=%s, credits=%d, teacher_id=%d)\n", item->id, item->title, item->code, item->credits, item->teacher_id);
}

/* Release the memory of the list. */
void course_list_free(course_list_t *list) {
    free(list->items);
    course_list_init(list);
}

/* A student. */
typedef struct {
    long id;
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    char email[NAME_LEN];
    time_t enrolled_at;
} student_t;

typedef struct {
    student_t *items;
    size_t count;
    size_t capacity;
} student_list_t;

/* Initialise an empty list. */
void student_list_init(student_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int student_list_push(student_list_t *list, const student_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        student_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the student with this id, or NULL. */
student_t *student_list_find(student_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one student on one line. */
void student_print(const student_t *item) {
    printf("Student(id=%ld, first_name=%s, last_name=%s, email=%s, enrolled_at=%ld)\n", item->id, item->first_name, item->last_name, item->email, (long)item->enrolled_at);
}

/* Release the memory of the list. */
void student_list_free(student_list_t *list) {
    free(list->items);
    student_list_init(list);
}

/* A enrollment. */
typedef struct {
    long id;
    int student_id;
    int course_id;
    double grade;
    bool completed;
} enrollment_t;

typedef struct {
    enrollment_t *items;
    size_t count;
    size_t capacity;
} enrollment_list_t;

/* Initialise an empty list. */
void enrollment_list_init(enrollment_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int enrollment_list_push(enrollment_list_t *list, const enrollment_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        enrollment_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the enrollment with this id, or NULL. */
enrollment_t *enrollment_list_find(enrollment_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one enrollment on one line. */
void enrollment_print(const enrollment_t *item) {
    printf("Enrollment(id=%ld, student_id=%d, course_id=%d, grade=%.2f, completed=%s)\n", item->id, item->student_id, item->course_id, item->grade, item->completed ? "true" : "false");
}

/* Release the memory of the list. */
void enrollment_list_free(enrollment_list_t *list) {
    free(list->items);
    enrollment_list_init(list);
}

/* A room. */
typedef struct {
    long id;
    char number[NAME_LEN];
    int floor;
    int beds;
    double price_per_night;
} room_t;

typedef struct {
    room_t *items;
    size_t count;
    size_t capacity;
} room_list_t;

/* Initialise an empty list. */
void room_list_init(room_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int room_list_push(room_list_t *list, const room_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        room_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the room with this id, or NULL. */
room_t *room_list_find(room_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one room on one line. */
void room_print(const room_t *item) {
    printf("Room(id=%ld, number=%s, floor=%d, beds=%d, price_per_night=%.2f)\n", item->id, item->number, item->floor, item->beds, item->price_per_night);
}

/* Release the memory of the list. */
void room_list_free(room_list_t *list) {
    free(list->items);
    room_list_init(list);
}

/* A booking. */
typedef struct {
    long id;
    int room_id;
    char guest_name[NAME_LEN];
    time_t check_in;
    time_t check_out;
    bool paid;
} booking_t;

typedef struct {
    booking_t *items;
    size_t count;
    size_t capacity;
} booking_list_t;

/* Initialise an empty list. */
void booking_list_init(booking_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int booking_list_push(booking_list_t *list, const booking_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        booking_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the booking with this id, or NULL. */
booking_t *booking_list_find(booking_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one booking on one line. */
void booking_print(const booking_t *item) {
    printf("Booking(id=%ld, room_id=%d, guest_name=%s, check_in=%ld, check_out=%ld, paid=%s)\n", item->id, item->room_id, item->guest_name, (long)item->check_in, (long)item->check_out, item->paid ? "true" : "false");
}

/* Release the memory of the list. */
void booking_list_free(booking_list_t *list) {
    free(list->items);
    booking_list_init(list);
}

/* A subscription. */
typedef struct {
    long id;
    int user_id;
    char plan[NAME_LEN];
    double price;
    time_t renews_at;
    bool is_active;
} subscription_t;

typedef struct {
    subscription_t *items;
    size_t count;
    size_t capacity;
} subscription_list_t;

/* Initialise an empty list. */
void subscription_list_init(subscription_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int subscription_list_push(subscription_list_t *list, const subscription_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        subscription_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the subscription with this id, or NULL. */
subscription_t *subscription_list_find(subscription_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one subscription on one line. */
void subscription_print(const subscription_t *item) {
    printf("Subscription(id=%ld, user_id=%d, plan=%s, price=%.2f, renews_at=%ld, is_active=%s)\n", item->id, item->user_id, item->plan, item->price, (long)item->renews_at, item->is_active ? "true" : "false");
}

/* Release the memory of the list. */
void subscription_list_free(subscription_list_t *list) {
    free(list->items);
    subscription_list_init(list);
}

/* A notification. */
typedef struct {
    long id;
    int user_id;
    char title[NAME_LEN];
    char message[NAME_LEN];
    bool read;
    time_t sent_at;
} notification_t;

typedef struct {
    notification_t *items;
    size_t count;
    size_t capacity;
} notification_list_t;

/* Initialise an empty list. */
void notification_list_init(notification_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int notification_list_push(notification_list_t *list, const notification_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        notification_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the notification with this id, or NULL. */
notification_t *notification_list_find(notification_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one notification on one line. */
void notification_print(const notification_t *item) {
    printf("Notification(id=%ld, user_id=%d, title=%s, message=%s, read=%s, sent_at=%ld)\n", item->id, item->user_id, item->title, item->message, item->read ? "true" : "false", (long)item->sent_at);
}

/* Release the memory of the list. */
void notification_list_free(notification_list_t *list) {
    free(list->items);
    notification_list_init(list);
}

/* A session. */
typedef struct {
    long id;
    int user_id;
    char token[NAME_LEN];
    time_t expires_at;
    char ip_address[NAME_LEN];
} session_t;

typedef struct {
    session_t *items;
    size_t count;
    size_t capacity;
} session_list_t;

/* Initialise an empty list. */
void session_list_init(session_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int session_list_push(session_list_t *list, const session_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        session_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the session with this id, or NULL. */
session_t *session_list_find(session_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one session on one line. */
void session_print(const session_t *item) {
    printf("Session(id=%ld, user_id=%d, token=%s, expires_at=%ld, ip_address=%s)\n", item->id, item->user_id, item->token, (long)item->expires_at, item->ip_address);
}

/* Release the memory of the list. */
void session_list_free(session_list_t *list) {
    free(list->items);
    session_list_init(list);
}

/* A audit log. */
typedef struct {
    long id;
    int actor_id;
    char action[NAME_LEN];
    char target[NAME_LEN];
    time_t created_at;
} audit_log_t;

typedef struct {
    audit_log_t *items;
    size_t count;
    size_t capacity;
} audit_log_list_t;

/* Initialise an empty list. */
void audit_log_list_init(audit_log_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int audit_log_list_push(audit_log_list_t *list, const audit_log_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        audit_log_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the audit log with this id, or NULL. */
audit_log_t *audit_log_list_find(audit_log_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one audit log on one line. */
void audit_log_print(const audit_log_t *item) {
    printf("AuditLog(id=%ld, actor_id=%d, action=%s, target=%s, created_at=%ld)\n", item->id, item->actor_id, item->action, item->target, (long)item->created_at);
}

/* Release the memory of the list. */
void audit_log_list_free(audit_log_list_t *list) {
    free(list->items);
    audit_log_list_init(list);
}

/* A setting. */
typedef struct {
    long id;
    char key[NAME_LEN];
    char value[NAME_LEN];
    char description[NAME_LEN];
} setting_t;

typedef struct {
    setting_t *items;
    size_t count;
    size_t capacity;
} setting_list_t;

/* Initialise an empty list. */
void setting_list_init(setting_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int setting_list_push(setting_list_t *list, const setting_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        setting_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the setting with this id, or NULL. */
setting_t *setting_list_find(setting_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one setting on one line. */
void setting_print(const setting_t *item) {
    printf("Setting(id=%ld, key=%s, value=%s, description=%s)\n", item->id, item->key, item->value, item->description);
}

/* Release the memory of the list. */
void setting_list_free(setting_list_t *list) {
    free(list->items);
    setting_list_init(list);
}

/* A recipe. */
typedef struct {
    long id;
    char title[NAME_LEN];
    int servings;
    int prep_minutes;
    int cook_minutes;
    bool vegetarian;
} recipe_t;

typedef struct {
    recipe_t *items;
    size_t count;
    size_t capacity;
} recipe_list_t;

/* Initialise an empty list. */
void recipe_list_init(recipe_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int recipe_list_push(recipe_list_t *list, const recipe_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        recipe_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the recipe with this id, or NULL. */
recipe_t *recipe_list_find(recipe_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one recipe on one line. */
void recipe_print(const recipe_t *item) {
    printf("Recipe(id=%ld, title=%s, servings=%d, prep_minutes=%d, cook_minutes=%d, vegetarian=%s)\n", item->id, item->title, item->servings, item->prep_minutes, item->cook_minutes, item->vegetarian ? "true" : "false");
}

/* Release the memory of the list. */
void recipe_list_free(recipe_list_t *list) {
    free(list->items);
    recipe_list_init(list);
}

/* A ingredient. */
typedef struct {
    long id;
    int recipe_id;
    char name[NAME_LEN];
    double amount;
    char unit[NAME_LEN];
} ingredient_t;

typedef struct {
    ingredient_t *items;
    size_t count;
    size_t capacity;
} ingredient_list_t;

/* Initialise an empty list. */
void ingredient_list_init(ingredient_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int ingredient_list_push(ingredient_list_t *list, const ingredient_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        ingredient_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the ingredient with this id, or NULL. */
ingredient_t *ingredient_list_find(ingredient_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one ingredient on one line. */
void ingredient_print(const ingredient_t *item) {
    printf("Ingredient(id=%ld, recipe_id=%d, name=%s, amount=%.2f, unit=%s)\n", item->id, item->recipe_id, item->name, item->amount, item->unit);
}

/* Release the memory of the list. */
void ingredient_list_free(ingredient_list_t *list) {
    free(list->items);
    ingredient_list_init(list);
}

/* A playlist. */
typedef struct {
    long id;
    char name[NAME_LEN];
    int owner_id;
    bool is_public;
    int track_count;
} playlist_t;

typedef struct {
    playlist_t *items;
    size_t count;
    size_t capacity;
} playlist_list_t;

/* Initialise an empty list. */
void playlist_list_init(playlist_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int playlist_list_push(playlist_list_t *list, const playlist_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        playlist_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the playlist with this id, or NULL. */
playlist_t *playlist_list_find(playlist_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one playlist on one line. */
void playlist_print(const playlist_t *item) {
    printf("Playlist(id=%ld, name=%s, owner_id=%d, is_public=%s, track_count=%d)\n", item->id, item->name, item->owner_id, item->is_public ? "true" : "false", item->track_count);
}

/* Release the memory of the list. */
void playlist_list_free(playlist_list_t *list) {
    free(list->items);
    playlist_list_init(list);
}

/* A track. */
typedef struct {
    long id;
    char title[NAME_LEN];
    char artist[NAME_LEN];
    char album[NAME_LEN];
    int duration_seconds;
} track_t;

typedef struct {
    track_t *items;
    size_t count;
    size_t capacity;
} track_list_t;

/* Initialise an empty list. */
void track_list_init(track_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int track_list_push(track_list_t *list, const track_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        track_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the track with this id, or NULL. */
track_t *track_list_find(track_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one track on one line. */
void track_print(const track_t *item) {
    printf("Track(id=%ld, title=%s, artist=%s, album=%s, duration_seconds=%d)\n", item->id, item->title, item->artist, item->album, item->duration_seconds);
}

/* Release the memory of the list. */
void track_list_free(track_list_t *list) {
    free(list->items);
    track_list_init(list);
}

/* A device. */
typedef struct {
    long id;
    char name[NAME_LEN];
    char serial_number[NAME_LEN];
    char firmware[NAME_LEN];
    bool online;
    time_t last_seen;
} device_t;

typedef struct {
    device_t *items;
    size_t count;
    size_t capacity;
} device_list_t;

/* Initialise an empty list. */
void device_list_init(device_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

/* Append a copy of item, growing the array when it is full. Returns 0 on success, -1 when out of memory. */
int device_list_push(device_list_t *list, const device_t *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        device_t *items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = *item;
    return 0;
}

/* Return the device with this id, or NULL. */
device_t *device_list_find(device_list_t *list, long id) {
    for (size_t i = 0; i < list->count; ++i) {
        if (list->items[i].id == id) {
            return &list->items[i];
        }
    }
    return NULL;
}

/* Print one device on one line. */
void device_print(const device_t *item) {
    printf("Device(id=%ld, name=%s, serial_number=%s, firmware=%s, online=%s, last_seen=%ld)\n", item->id, item->name, item->serial_number, item->firmware, item->online ? "true" : "false", (long)item->last_seen);
}

/* Release the memory of the list. */
void device_list_free(device_list_t *list) {
    free(list->items);
    device_list_init(list);
}
