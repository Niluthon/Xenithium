package com.example.store.model;

import java.time.LocalDateTime;
import java.util.Objects;

/** A user. */
public class User {
    private Long id;
    private String username;
    private String email;
    private String fullName;
    private boolean isActive;
    private LocalDateTime createdAt;

    public User() {
    }

    public User(Long id, String username, String email, String fullName, boolean isActive, LocalDateTime createdAt) {
        this.id = id;
        this.username = username;
        this.email = email;
        this.fullName = fullName;
        this.isActive = isActive;
        this.createdAt = createdAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getFullName() {
        return fullName;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
    }

    public boolean isIsActive() {
        return isActive;
    }

    public void setIsActive(boolean isActive) {
        this.isActive = isActive;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        User other = (User) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id + ", " +
                "username=" + username + ", " +
                "email=" + email + ", " +
                "fullName=" + fullName + ", " +
                "isActive=" + isActive + ", " +
                "createdAt=" + createdAt +
                "}";
    }
}

/** A product. */
public class Product {
    private Long id;
    private String sku;
    private String name;
    private String description;
    private double price;
    private int stock;

    public Product() {
    }

    public Product(Long id, String sku, String name, String description, double price, int stock) {
        this.id = id;
        this.sku = sku;
        this.name = name;
        this.description = description;
        this.price = price;
        this.stock = stock;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getSku() {
        return sku;
    }

    public void setSku(String sku) {
        this.sku = sku;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    public int getStock() {
        return stock;
    }

    public void setStock(int stock) {
        this.stock = stock;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Product other = (Product) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Product{" +
                "id=" + id + ", " +
                "sku=" + sku + ", " +
                "name=" + name + ", " +
                "description=" + description + ", " +
                "price=" + price + ", " +
                "stock=" + stock +
                "}";
    }
}

/** A order. */
public class Order {
    private Long id;
    private int customerId;
    private String status;
    private double total;
    private String currency;
    private LocalDateTime placedAt;

    public Order() {
    }

    public Order(Long id, int customerId, String status, double total, String currency, LocalDateTime placedAt) {
        this.id = id;
        this.customerId = customerId;
        this.status = status;
        this.total = total;
        this.currency = currency;
        this.placedAt = placedAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getCustomerId() {
        return customerId;
    }

    public void setCustomerId(int customerId) {
        this.customerId = customerId;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public double getTotal() {
        return total;
    }

    public void setTotal(double total) {
        this.total = total;
    }

    public String getCurrency() {
        return currency;
    }

    public void setCurrency(String currency) {
        this.currency = currency;
    }

    public LocalDateTime getPlacedAt() {
        return placedAt;
    }

    public void setPlacedAt(LocalDateTime placedAt) {
        this.placedAt = placedAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Order other = (Order) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Order{" +
                "id=" + id + ", " +
                "customerId=" + customerId + ", " +
                "status=" + status + ", " +
                "total=" + total + ", " +
                "currency=" + currency + ", " +
                "placedAt=" + placedAt +
                "}";
    }
}

/** A order item. */
public class OrderItem {
    private Long id;
    private int orderId;
    private int productId;
    private int quantity;
    private double unitPrice;

    public OrderItem() {
    }

    public OrderItem(Long id, int orderId, int productId, int quantity, double unitPrice) {
        this.id = id;
        this.orderId = orderId;
        this.productId = productId;
        this.quantity = quantity;
        this.unitPrice = unitPrice;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getOrderId() {
        return orderId;
    }

    public void setOrderId(int orderId) {
        this.orderId = orderId;
    }

    public int getProductId() {
        return productId;
    }

    public void setProductId(int productId) {
        this.productId = productId;
    }

    public int getQuantity() {
        return quantity;
    }

    public void setQuantity(int quantity) {
        this.quantity = quantity;
    }

    public double getUnitPrice() {
        return unitPrice;
    }

    public void setUnitPrice(double unitPrice) {
        this.unitPrice = unitPrice;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        OrderItem other = (OrderItem) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "OrderItem{" +
                "id=" + id + ", " +
                "orderId=" + orderId + ", " +
                "productId=" + productId + ", " +
                "quantity=" + quantity + ", " +
                "unitPrice=" + unitPrice +
                "}";
    }
}

/** A customer. */
public class Customer {
    private Long id;
    private String firstName;
    private String lastName;
    private String email;
    private String phone;
    private String country;

    public Customer() {
    }

    public Customer(Long id, String firstName, String lastName, String email, String phone, String country) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.phone = phone;
        this.country = country;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }

    public String getCountry() {
        return country;
    }

    public void setCountry(String country) {
        this.country = country;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Customer other = (Customer) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Customer{" +
                "id=" + id + ", " +
                "firstName=" + firstName + ", " +
                "lastName=" + lastName + ", " +
                "email=" + email + ", " +
                "phone=" + phone + ", " +
                "country=" + country +
                "}";
    }
}

/** A invoice. */
public class Invoice {
    private Long id;
    private int orderId;
    private String number;
    private double amount;
    private LocalDateTime dueDate;
    private boolean paid;

    public Invoice() {
    }

    public Invoice(Long id, int orderId, String number, double amount, LocalDateTime dueDate, boolean paid) {
        this.id = id;
        this.orderId = orderId;
        this.number = number;
        this.amount = amount;
        this.dueDate = dueDate;
        this.paid = paid;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getOrderId() {
        return orderId;
    }

    public void setOrderId(int orderId) {
        this.orderId = orderId;
    }

    public String getNumber() {
        return number;
    }

    public void setNumber(String number) {
        this.number = number;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public LocalDateTime getDueDate() {
        return dueDate;
    }

    public void setDueDate(LocalDateTime dueDate) {
        this.dueDate = dueDate;
    }

    public boolean isPaid() {
        return paid;
    }

    public void setPaid(boolean paid) {
        this.paid = paid;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Invoice other = (Invoice) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Invoice{" +
                "id=" + id + ", " +
                "orderId=" + orderId + ", " +
                "number=" + number + ", " +
                "amount=" + amount + ", " +
                "dueDate=" + dueDate + ", " +
                "paid=" + paid +
                "}";
    }
}

/** A payment. */
public class Payment {
    private Long id;
    private int invoiceId;
    private String method;
    private double amount;
    private String reference;
    private LocalDateTime receivedAt;

    public Payment() {
    }

    public Payment(Long id, int invoiceId, String method, double amount, String reference, LocalDateTime receivedAt) {
        this.id = id;
        this.invoiceId = invoiceId;
        this.method = method;
        this.amount = amount;
        this.reference = reference;
        this.receivedAt = receivedAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getInvoiceId() {
        return invoiceId;
    }

    public void setInvoiceId(int invoiceId) {
        this.invoiceId = invoiceId;
    }

    public String getMethod() {
        return method;
    }

    public void setMethod(String method) {
        this.method = method;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public String getReference() {
        return reference;
    }

    public void setReference(String reference) {
        this.reference = reference;
    }

    public LocalDateTime getReceivedAt() {
        return receivedAt;
    }

    public void setReceivedAt(LocalDateTime receivedAt) {
        this.receivedAt = receivedAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Payment other = (Payment) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Payment{" +
                "id=" + id + ", " +
                "invoiceId=" + invoiceId + ", " +
                "method=" + method + ", " +
                "amount=" + amount + ", " +
                "reference=" + reference + ", " +
                "receivedAt=" + receivedAt +
                "}";
    }
}

/** A category. */
public class Category {
    private Long id;
    private String name;
    private String slug;
    private int parentId;
    private int position;

    public Category() {
    }

    public Category(Long id, String name, String slug, int parentId, int position) {
        this.id = id;
        this.name = name;
        this.slug = slug;
        this.parentId = parentId;
        this.position = position;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSlug() {
        return slug;
    }

    public void setSlug(String slug) {
        this.slug = slug;
    }

    public int getParentId() {
        return parentId;
    }

    public void setParentId(int parentId) {
        this.parentId = parentId;
    }

    public int getPosition() {
        return position;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Category other = (Category) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Category{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "slug=" + slug + ", " +
                "parentId=" + parentId + ", " +
                "position=" + position +
                "}";
    }
}

/** A review. */
public class Review {
    private Long id;
    private int productId;
    private int userId;
    private int rating;
    private String title;
    private String body;

    public Review() {
    }

    public Review(Long id, int productId, int userId, int rating, String title, String body) {
        this.id = id;
        this.productId = productId;
        this.userId = userId;
        this.rating = rating;
        this.title = title;
        this.body = body;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getProductId() {
        return productId;
    }

    public void setProductId(int productId) {
        this.productId = productId;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public int getRating() {
        return rating;
    }

    public void setRating(int rating) {
        this.rating = rating;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getBody() {
        return body;
    }

    public void setBody(String body) {
        this.body = body;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Review other = (Review) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Review{" +
                "id=" + id + ", " +
                "productId=" + productId + ", " +
                "userId=" + userId + ", " +
                "rating=" + rating + ", " +
                "title=" + title + ", " +
                "body=" + body +
                "}";
    }
}

/** A address. */
public class Address {
    private Long id;
    private int customerId;
    private String street;
    private String city;
    private String postalCode;
    private String country;

    public Address() {
    }

    public Address(Long id, int customerId, String street, String city, String postalCode, String country) {
        this.id = id;
        this.customerId = customerId;
        this.street = street;
        this.city = city;
        this.postalCode = postalCode;
        this.country = country;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getCustomerId() {
        return customerId;
    }

    public void setCustomerId(int customerId) {
        this.customerId = customerId;
    }

    public String getStreet() {
        return street;
    }

    public void setStreet(String street) {
        this.street = street;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public String getPostalCode() {
        return postalCode;
    }

    public void setPostalCode(String postalCode) {
        this.postalCode = postalCode;
    }

    public String getCountry() {
        return country;
    }

    public void setCountry(String country) {
        this.country = country;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Address other = (Address) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Address{" +
                "id=" + id + ", " +
                "customerId=" + customerId + ", " +
                "street=" + street + ", " +
                "city=" + city + ", " +
                "postalCode=" + postalCode + ", " +
                "country=" + country +
                "}";
    }
}

/** A shipment. */
public class Shipment {
    private Long id;
    private int orderId;
    private String carrier;
    private String trackingNumber;
    private LocalDateTime shippedAt;
    private boolean delivered;

    public Shipment() {
    }

    public Shipment(Long id, int orderId, String carrier, String trackingNumber, LocalDateTime shippedAt, boolean delivered) {
        this.id = id;
        this.orderId = orderId;
        this.carrier = carrier;
        this.trackingNumber = trackingNumber;
        this.shippedAt = shippedAt;
        this.delivered = delivered;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getOrderId() {
        return orderId;
    }

    public void setOrderId(int orderId) {
        this.orderId = orderId;
    }

    public String getCarrier() {
        return carrier;
    }

    public void setCarrier(String carrier) {
        this.carrier = carrier;
    }

    public String getTrackingNumber() {
        return trackingNumber;
    }

    public void setTrackingNumber(String trackingNumber) {
        this.trackingNumber = trackingNumber;
    }

    public LocalDateTime getShippedAt() {
        return shippedAt;
    }

    public void setShippedAt(LocalDateTime shippedAt) {
        this.shippedAt = shippedAt;
    }

    public boolean isDelivered() {
        return delivered;
    }

    public void setDelivered(boolean delivered) {
        this.delivered = delivered;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Shipment other = (Shipment) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Shipment{" +
                "id=" + id + ", " +
                "orderId=" + orderId + ", " +
                "carrier=" + carrier + ", " +
                "trackingNumber=" + trackingNumber + ", " +
                "shippedAt=" + shippedAt + ", " +
                "delivered=" + delivered +
                "}";
    }
}

/** A warehouse. */
public class Warehouse {
    private Long id;
    private String name;
    private String city;
    private int capacity;
    private boolean isOpen;

    public Warehouse() {
    }

    public Warehouse(Long id, String name, String city, int capacity, boolean isOpen) {
        this.id = id;
        this.name = name;
        this.city = city;
        this.capacity = capacity;
        this.isOpen = isOpen;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public int getCapacity() {
        return capacity;
    }

    public void setCapacity(int capacity) {
        this.capacity = capacity;
    }

    public boolean isIsOpen() {
        return isOpen;
    }

    public void setIsOpen(boolean isOpen) {
        this.isOpen = isOpen;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Warehouse other = (Warehouse) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Warehouse{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "city=" + city + ", " +
                "capacity=" + capacity + ", " +
                "isOpen=" + isOpen +
                "}";
    }
}

/** A supplier. */
public class Supplier {
    private Long id;
    private String name;
    private String contactEmail;
    private String phone;
    private String country;
    private int rating;

    public Supplier() {
    }

    public Supplier(Long id, String name, String contactEmail, String phone, String country, int rating) {
        this.id = id;
        this.name = name;
        this.contactEmail = contactEmail;
        this.phone = phone;
        this.country = country;
        this.rating = rating;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getContactEmail() {
        return contactEmail;
    }

    public void setContactEmail(String contactEmail) {
        this.contactEmail = contactEmail;
    }

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }

    public String getCountry() {
        return country;
    }

    public void setCountry(String country) {
        this.country = country;
    }

    public int getRating() {
        return rating;
    }

    public void setRating(int rating) {
        this.rating = rating;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Supplier other = (Supplier) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Supplier{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "contactEmail=" + contactEmail + ", " +
                "phone=" + phone + ", " +
                "country=" + country + ", " +
                "rating=" + rating +
                "}";
    }
}

/** A coupon. */
public class Coupon {
    private Long id;
    private String code;
    private int discountPercent;
    private LocalDateTime validUntil;
    private int maxUses;
    private boolean isActive;

    public Coupon() {
    }

    public Coupon(Long id, String code, int discountPercent, LocalDateTime validUntil, int maxUses, boolean isActive) {
        this.id = id;
        this.code = code;
        this.discountPercent = discountPercent;
        this.validUntil = validUntil;
        this.maxUses = maxUses;
        this.isActive = isActive;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public int getDiscountPercent() {
        return discountPercent;
    }

    public void setDiscountPercent(int discountPercent) {
        this.discountPercent = discountPercent;
    }

    public LocalDateTime getValidUntil() {
        return validUntil;
    }

    public void setValidUntil(LocalDateTime validUntil) {
        this.validUntil = validUntil;
    }

    public int getMaxUses() {
        return maxUses;
    }

    public void setMaxUses(int maxUses) {
        this.maxUses = maxUses;
    }

    public boolean isIsActive() {
        return isActive;
    }

    public void setIsActive(boolean isActive) {
        this.isActive = isActive;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Coupon other = (Coupon) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Coupon{" +
                "id=" + id + ", " +
                "code=" + code + ", " +
                "discountPercent=" + discountPercent + ", " +
                "validUntil=" + validUntil + ", " +
                "maxUses=" + maxUses + ", " +
                "isActive=" + isActive +
                "}";
    }
}

/** A employee. */
public class Employee {
    private Long id;
    private String firstName;
    private String lastName;
    private String email;
    private String department;
    private double salary;

    public Employee() {
    }

    public Employee(Long id, String firstName, String lastName, String email, String department, double salary) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.department = department;
        this.salary = salary;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getDepartment() {
        return department;
    }

    public void setDepartment(String department) {
        this.department = department;
    }

    public double getSalary() {
        return salary;
    }

    public void setSalary(double salary) {
        this.salary = salary;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Employee other = (Employee) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Employee{" +
                "id=" + id + ", " +
                "firstName=" + firstName + ", " +
                "lastName=" + lastName + ", " +
                "email=" + email + ", " +
                "department=" + department + ", " +
                "salary=" + salary +
                "}";
    }
}

/** A department. */
public class Department {
    private Long id;
    private String name;
    private String code;
    private int managerId;
    private double budget;

    public Department() {
    }

    public Department(Long id, String name, String code, int managerId, double budget) {
        this.id = id;
        this.name = name;
        this.code = code;
        this.managerId = managerId;
        this.budget = budget;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public int getManagerId() {
        return managerId;
    }

    public void setManagerId(int managerId) {
        this.managerId = managerId;
    }

    public double getBudget() {
        return budget;
    }

    public void setBudget(double budget) {
        this.budget = budget;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Department other = (Department) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Department{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "code=" + code + ", " +
                "managerId=" + managerId + ", " +
                "budget=" + budget +
                "}";
    }
}

/** A project. */
public class Project {
    private Long id;
    private String name;
    private String description;
    private LocalDateTime startDate;
    private LocalDateTime endDate;
    private double budget;

    public Project() {
    }

    public Project(Long id, String name, String description, LocalDateTime startDate, LocalDateTime endDate, double budget) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.startDate = startDate;
        this.endDate = endDate;
        this.budget = budget;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public LocalDateTime getStartDate() {
        return startDate;
    }

    public void setStartDate(LocalDateTime startDate) {
        this.startDate = startDate;
    }

    public LocalDateTime getEndDate() {
        return endDate;
    }

    public void setEndDate(LocalDateTime endDate) {
        this.endDate = endDate;
    }

    public double getBudget() {
        return budget;
    }

    public void setBudget(double budget) {
        this.budget = budget;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Project other = (Project) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Project{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "description=" + description + ", " +
                "startDate=" + startDate + ", " +
                "endDate=" + endDate + ", " +
                "budget=" + budget +
                "}";
    }
}

/** A task. */
public class Task {
    private Long id;
    private int projectId;
    private String title;
    private String status;
    private int priority;
    private LocalDateTime dueDate;

    public Task() {
    }

    public Task(Long id, int projectId, String title, String status, int priority, LocalDateTime dueDate) {
        this.id = id;
        this.projectId = projectId;
        this.title = title;
        this.status = status;
        this.priority = priority;
        this.dueDate = dueDate;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getProjectId() {
        return projectId;
    }

    public void setProjectId(int projectId) {
        this.projectId = projectId;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public int getPriority() {
        return priority;
    }

    public void setPriority(int priority) {
        this.priority = priority;
    }

    public LocalDateTime getDueDate() {
        return dueDate;
    }

    public void setDueDate(LocalDateTime dueDate) {
        this.dueDate = dueDate;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Task other = (Task) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Task{" +
                "id=" + id + ", " +
                "projectId=" + projectId + ", " +
                "title=" + title + ", " +
                "status=" + status + ", " +
                "priority=" + priority + ", " +
                "dueDate=" + dueDate +
                "}";
    }
}

/** A comment. */
public class Comment {
    private Long id;
    private int taskId;
    private int authorId;
    private String body;
    private LocalDateTime createdAt;

    public Comment() {
    }

    public Comment(Long id, int taskId, int authorId, String body, LocalDateTime createdAt) {
        this.id = id;
        this.taskId = taskId;
        this.authorId = authorId;
        this.body = body;
        this.createdAt = createdAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getTaskId() {
        return taskId;
    }

    public void setTaskId(int taskId) {
        this.taskId = taskId;
    }

    public int getAuthorId() {
        return authorId;
    }

    public void setAuthorId(int authorId) {
        this.authorId = authorId;
    }

    public String getBody() {
        return body;
    }

    public void setBody(String body) {
        this.body = body;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Comment other = (Comment) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Comment{" +
                "id=" + id + ", " +
                "taskId=" + taskId + ", " +
                "authorId=" + authorId + ", " +
                "body=" + body + ", " +
                "createdAt=" + createdAt +
                "}";
    }
}

/** A tag. */
public class Tag {
    private Long id;
    private String name;
    private String color;
    private int usageCount;

    public Tag() {
    }

    public Tag(Long id, String name, String color, int usageCount) {
        this.id = id;
        this.name = name;
        this.color = color;
        this.usageCount = usageCount;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    public int getUsageCount() {
        return usageCount;
    }

    public void setUsageCount(int usageCount) {
        this.usageCount = usageCount;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Tag other = (Tag) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Tag{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "color=" + color + ", " +
                "usageCount=" + usageCount +
                "}";
    }
}

/** A article. */
public class Article {
    private Long id;
    private String title;
    private String slug;
    private String body;
    private int authorId;
    private boolean published;

    public Article() {
    }

    public Article(Long id, String title, String slug, String body, int authorId, boolean published) {
        this.id = id;
        this.title = title;
        this.slug = slug;
        this.body = body;
        this.authorId = authorId;
        this.published = published;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getSlug() {
        return slug;
    }

    public void setSlug(String slug) {
        this.slug = slug;
    }

    public String getBody() {
        return body;
    }

    public void setBody(String body) {
        this.body = body;
    }

    public int getAuthorId() {
        return authorId;
    }

    public void setAuthorId(int authorId) {
        this.authorId = authorId;
    }

    public boolean isPublished() {
        return published;
    }

    public void setPublished(boolean published) {
        this.published = published;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Article other = (Article) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Article{" +
                "id=" + id + ", " +
                "title=" + title + ", " +
                "slug=" + slug + ", " +
                "body=" + body + ", " +
                "authorId=" + authorId + ", " +
                "published=" + published +
                "}";
    }
}

/** A author. */
public class Author {
    private Long id;
    private String name;
    private String bio;
    private String email;
    private String website;

    public Author() {
    }

    public Author(Long id, String name, String bio, String email, String website) {
        this.id = id;
        this.name = name;
        this.bio = bio;
        this.email = email;
        this.website = website;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getBio() {
        return bio;
    }

    public void setBio(String bio) {
        this.bio = bio;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getWebsite() {
        return website;
    }

    public void setWebsite(String website) {
        this.website = website;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Author other = (Author) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Author{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "bio=" + bio + ", " +
                "email=" + email + ", " +
                "website=" + website +
                "}";
    }
}

/** A book. */
public class Book {
    private Long id;
    private String title;
    private String isbn;
    private int authorId;
    private int year;
    private int pages;

    public Book() {
    }

    public Book(Long id, String title, String isbn, int authorId, int year, int pages) {
        this.id = id;
        this.title = title;
        this.isbn = isbn;
        this.authorId = authorId;
        this.year = year;
        this.pages = pages;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getIsbn() {
        return isbn;
    }

    public void setIsbn(String isbn) {
        this.isbn = isbn;
    }

    public int getAuthorId() {
        return authorId;
    }

    public void setAuthorId(int authorId) {
        this.authorId = authorId;
    }

    public int getYear() {
        return year;
    }

    public void setYear(int year) {
        this.year = year;
    }

    public int getPages() {
        return pages;
    }

    public void setPages(int pages) {
        this.pages = pages;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Book other = (Book) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Book{" +
                "id=" + id + ", " +
                "title=" + title + ", " +
                "isbn=" + isbn + ", " +
                "authorId=" + authorId + ", " +
                "year=" + year + ", " +
                "pages=" + pages +
                "}";
    }
}

/** A library. */
public class Library {
    private Long id;
    private String name;
    private String city;
    private int openingHour;
    private int closingHour;

    public Library() {
    }

    public Library(Long id, String name, String city, int openingHour, int closingHour) {
        this.id = id;
        this.name = name;
        this.city = city;
        this.openingHour = openingHour;
        this.closingHour = closingHour;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public int getOpeningHour() {
        return openingHour;
    }

    public void setOpeningHour(int openingHour) {
        this.openingHour = openingHour;
    }

    public int getClosingHour() {
        return closingHour;
    }

    public void setClosingHour(int closingHour) {
        this.closingHour = closingHour;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Library other = (Library) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Library{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "city=" + city + ", " +
                "openingHour=" + openingHour + ", " +
                "closingHour=" + closingHour +
                "}";
    }
}

/** A loan. */
public class Loan {
    private Long id;
    private int bookId;
    private int memberId;
    private LocalDateTime borrowedAt;
    private LocalDateTime dueDate;
    private boolean returned;

    public Loan() {
    }

    public Loan(Long id, int bookId, int memberId, LocalDateTime borrowedAt, LocalDateTime dueDate, boolean returned) {
        this.id = id;
        this.bookId = bookId;
        this.memberId = memberId;
        this.borrowedAt = borrowedAt;
        this.dueDate = dueDate;
        this.returned = returned;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getBookId() {
        return bookId;
    }

    public void setBookId(int bookId) {
        this.bookId = bookId;
    }

    public int getMemberId() {
        return memberId;
    }

    public void setMemberId(int memberId) {
        this.memberId = memberId;
    }

    public LocalDateTime getBorrowedAt() {
        return borrowedAt;
    }

    public void setBorrowedAt(LocalDateTime borrowedAt) {
        this.borrowedAt = borrowedAt;
    }

    public LocalDateTime getDueDate() {
        return dueDate;
    }

    public void setDueDate(LocalDateTime dueDate) {
        this.dueDate = dueDate;
    }

    public boolean isReturned() {
        return returned;
    }

    public void setReturned(boolean returned) {
        this.returned = returned;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Loan other = (Loan) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Loan{" +
                "id=" + id + ", " +
                "bookId=" + bookId + ", " +
                "memberId=" + memberId + ", " +
                "borrowedAt=" + borrowedAt + ", " +
                "dueDate=" + dueDate + ", " +
                "returned=" + returned +
                "}";
    }
}

/** A member. */
public class Member {
    private Long id;
    private String firstName;
    private String lastName;
    private String email;
    private LocalDateTime joinedAt;

    public Member() {
    }

    public Member(Long id, String firstName, String lastName, String email, LocalDateTime joinedAt) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.joinedAt = joinedAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public LocalDateTime getJoinedAt() {
        return joinedAt;
    }

    public void setJoinedAt(LocalDateTime joinedAt) {
        this.joinedAt = joinedAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Member other = (Member) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Member{" +
                "id=" + id + ", " +
                "firstName=" + firstName + ", " +
                "lastName=" + lastName + ", " +
                "email=" + email + ", " +
                "joinedAt=" + joinedAt +
                "}";
    }
}

/** A event. */
public class Event {
    private Long id;
    private String title;
    private String location;
    private LocalDateTime startsAt;
    private LocalDateTime endsAt;
    private int capacity;

    public Event() {
    }

    public Event(Long id, String title, String location, LocalDateTime startsAt, LocalDateTime endsAt, int capacity) {
        this.id = id;
        this.title = title;
        this.location = location;
        this.startsAt = startsAt;
        this.endsAt = endsAt;
        this.capacity = capacity;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public LocalDateTime getStartsAt() {
        return startsAt;
    }

    public void setStartsAt(LocalDateTime startsAt) {
        this.startsAt = startsAt;
    }

    public LocalDateTime getEndsAt() {
        return endsAt;
    }

    public void setEndsAt(LocalDateTime endsAt) {
        this.endsAt = endsAt;
    }

    public int getCapacity() {
        return capacity;
    }

    public void setCapacity(int capacity) {
        this.capacity = capacity;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Event other = (Event) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Event{" +
                "id=" + id + ", " +
                "title=" + title + ", " +
                "location=" + location + ", " +
                "startsAt=" + startsAt + ", " +
                "endsAt=" + endsAt + ", " +
                "capacity=" + capacity +
                "}";
    }
}

/** A ticket. */
public class Ticket {
    private Long id;
    private int eventId;
    private String holderName;
    private String seat;
    private double price;
    private boolean checkedIn;

    public Ticket() {
    }

    public Ticket(Long id, int eventId, String holderName, String seat, double price, boolean checkedIn) {
        this.id = id;
        this.eventId = eventId;
        this.holderName = holderName;
        this.seat = seat;
        this.price = price;
        this.checkedIn = checkedIn;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getEventId() {
        return eventId;
    }

    public void setEventId(int eventId) {
        this.eventId = eventId;
    }

    public String getHolderName() {
        return holderName;
    }

    public void setHolderName(String holderName) {
        this.holderName = holderName;
    }

    public String getSeat() {
        return seat;
    }

    public void setSeat(String seat) {
        this.seat = seat;
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    public boolean isCheckedIn() {
        return checkedIn;
    }

    public void setCheckedIn(boolean checkedIn) {
        this.checkedIn = checkedIn;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Ticket other = (Ticket) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Ticket{" +
                "id=" + id + ", " +
                "eventId=" + eventId + ", " +
                "holderName=" + holderName + ", " +
                "seat=" + seat + ", " +
                "price=" + price + ", " +
                "checkedIn=" + checkedIn +
                "}";
    }
}

/** A venue. */
public class Venue {
    private Long id;
    private String name;
    private String address;
    private String city;
    private int capacity;

    public Venue() {
    }

    public Venue(Long id, String name, String address, String city, int capacity) {
        this.id = id;
        this.name = name;
        this.address = address;
        this.city = city;
        this.capacity = capacity;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public int getCapacity() {
        return capacity;
    }

    public void setCapacity(int capacity) {
        this.capacity = capacity;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Venue other = (Venue) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Venue{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "address=" + address + ", " +
                "city=" + city + ", " +
                "capacity=" + capacity +
                "}";
    }
}

/** A vehicle. */
public class Vehicle {
    private Long id;
    private String plate;
    private String make;
    private String model;
    private int year;
    private int mileage;

    public Vehicle() {
    }

    public Vehicle(Long id, String plate, String make, String model, int year, int mileage) {
        this.id = id;
        this.plate = plate;
        this.make = make;
        this.model = model;
        this.year = year;
        this.mileage = mileage;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getPlate() {
        return plate;
    }

    public void setPlate(String plate) {
        this.plate = plate;
    }

    public String getMake() {
        return make;
    }

    public void setMake(String make) {
        this.make = make;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public int getYear() {
        return year;
    }

    public void setYear(int year) {
        this.year = year;
    }

    public int getMileage() {
        return mileage;
    }

    public void setMileage(int mileage) {
        this.mileage = mileage;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Vehicle other = (Vehicle) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Vehicle{" +
                "id=" + id + ", " +
                "plate=" + plate + ", " +
                "make=" + make + ", " +
                "model=" + model + ", " +
                "year=" + year + ", " +
                "mileage=" + mileage +
                "}";
    }
}

/** A driver. */
public class Driver {
    private Long id;
    private String firstName;
    private String lastName;
    private String licenseNumber;
    private double rating;

    public Driver() {
    }

    public Driver(Long id, String firstName, String lastName, String licenseNumber, double rating) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.licenseNumber = licenseNumber;
        this.rating = rating;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getLicenseNumber() {
        return licenseNumber;
    }

    public void setLicenseNumber(String licenseNumber) {
        this.licenseNumber = licenseNumber;
    }

    public double getRating() {
        return rating;
    }

    public void setRating(double rating) {
        this.rating = rating;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Driver other = (Driver) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Driver{" +
                "id=" + id + ", " +
                "firstName=" + firstName + ", " +
                "lastName=" + lastName + ", " +
                "licenseNumber=" + licenseNumber + ", " +
                "rating=" + rating +
                "}";
    }
}

/** A trip. */
public class Trip {
    private Long id;
    private int vehicleId;
    private int driverId;
    private double distanceKm;
    private LocalDateTime startedAt;
    private double fare;

    public Trip() {
    }

    public Trip(Long id, int vehicleId, int driverId, double distanceKm, LocalDateTime startedAt, double fare) {
        this.id = id;
        this.vehicleId = vehicleId;
        this.driverId = driverId;
        this.distanceKm = distanceKm;
        this.startedAt = startedAt;
        this.fare = fare;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getVehicleId() {
        return vehicleId;
    }

    public void setVehicleId(int vehicleId) {
        this.vehicleId = vehicleId;
    }

    public int getDriverId() {
        return driverId;
    }

    public void setDriverId(int driverId) {
        this.driverId = driverId;
    }

    public double getDistanceKm() {
        return distanceKm;
    }

    public void setDistanceKm(double distanceKm) {
        this.distanceKm = distanceKm;
    }

    public LocalDateTime getStartedAt() {
        return startedAt;
    }

    public void setStartedAt(LocalDateTime startedAt) {
        this.startedAt = startedAt;
    }

    public double getFare() {
        return fare;
    }

    public void setFare(double fare) {
        this.fare = fare;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Trip other = (Trip) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Trip{" +
                "id=" + id + ", " +
                "vehicleId=" + vehicleId + ", " +
                "driverId=" + driverId + ", " +
                "distanceKm=" + distanceKm + ", " +
                "startedAt=" + startedAt + ", " +
                "fare=" + fare +
                "}";
    }
}

/** A patient. */
public class Patient {
    private Long id;
    private String firstName;
    private String lastName;
    private LocalDateTime birthDate;
    private String phone;

    public Patient() {
    }

    public Patient(Long id, String firstName, String lastName, LocalDateTime birthDate, String phone) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.birthDate = birthDate;
        this.phone = phone;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public LocalDateTime getBirthDate() {
        return birthDate;
    }

    public void setBirthDate(LocalDateTime birthDate) {
        this.birthDate = birthDate;
    }

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Patient other = (Patient) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Patient{" +
                "id=" + id + ", " +
                "firstName=" + firstName + ", " +
                "lastName=" + lastName + ", " +
                "birthDate=" + birthDate + ", " +
                "phone=" + phone +
                "}";
    }
}

/** A doctor. */
public class Doctor {
    private Long id;
    private String firstName;
    private String lastName;
    private String specialty;
    private String room;

    public Doctor() {
    }

    public Doctor(Long id, String firstName, String lastName, String specialty, String room) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.specialty = specialty;
        this.room = room;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getSpecialty() {
        return specialty;
    }

    public void setSpecialty(String specialty) {
        this.specialty = specialty;
    }

    public String getRoom() {
        return room;
    }

    public void setRoom(String room) {
        this.room = room;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Doctor other = (Doctor) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Doctor{" +
                "id=" + id + ", " +
                "firstName=" + firstName + ", " +
                "lastName=" + lastName + ", " +
                "specialty=" + specialty + ", " +
                "room=" + room +
                "}";
    }
}

/** A appointment. */
public class Appointment {
    private Long id;
    private int patientId;
    private int doctorId;
    private LocalDateTime scheduledAt;
    private String reason;
    private boolean confirmed;

    public Appointment() {
    }

    public Appointment(Long id, int patientId, int doctorId, LocalDateTime scheduledAt, String reason, boolean confirmed) {
        this.id = id;
        this.patientId = patientId;
        this.doctorId = doctorId;
        this.scheduledAt = scheduledAt;
        this.reason = reason;
        this.confirmed = confirmed;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getPatientId() {
        return patientId;
    }

    public void setPatientId(int patientId) {
        this.patientId = patientId;
    }

    public int getDoctorId() {
        return doctorId;
    }

    public void setDoctorId(int doctorId) {
        this.doctorId = doctorId;
    }

    public LocalDateTime getScheduledAt() {
        return scheduledAt;
    }

    public void setScheduledAt(LocalDateTime scheduledAt) {
        this.scheduledAt = scheduledAt;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public boolean isConfirmed() {
        return confirmed;
    }

    public void setConfirmed(boolean confirmed) {
        this.confirmed = confirmed;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Appointment other = (Appointment) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Appointment{" +
                "id=" + id + ", " +
                "patientId=" + patientId + ", " +
                "doctorId=" + doctorId + ", " +
                "scheduledAt=" + scheduledAt + ", " +
                "reason=" + reason + ", " +
                "confirmed=" + confirmed +
                "}";
    }
}

/** A course. */
public class Course {
    private Long id;
    private String title;
    private String code;
    private int credits;
    private int teacherId;

    public Course() {
    }

    public Course(Long id, String title, String code, int credits, int teacherId) {
        this.id = id;
        this.title = title;
        this.code = code;
        this.credits = credits;
        this.teacherId = teacherId;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public int getCredits() {
        return credits;
    }

    public void setCredits(int credits) {
        this.credits = credits;
    }

    public int getTeacherId() {
        return teacherId;
    }

    public void setTeacherId(int teacherId) {
        this.teacherId = teacherId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Course other = (Course) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Course{" +
                "id=" + id + ", " +
                "title=" + title + ", " +
                "code=" + code + ", " +
                "credits=" + credits + ", " +
                "teacherId=" + teacherId +
                "}";
    }
}

/** A student. */
public class Student {
    private Long id;
    private String firstName;
    private String lastName;
    private String email;
    private LocalDateTime enrolledAt;

    public Student() {
    }

    public Student(Long id, String firstName, String lastName, String email, LocalDateTime enrolledAt) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.enrolledAt = enrolledAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public LocalDateTime getEnrolledAt() {
        return enrolledAt;
    }

    public void setEnrolledAt(LocalDateTime enrolledAt) {
        this.enrolledAt = enrolledAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Student other = (Student) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Student{" +
                "id=" + id + ", " +
                "firstName=" + firstName + ", " +
                "lastName=" + lastName + ", " +
                "email=" + email + ", " +
                "enrolledAt=" + enrolledAt +
                "}";
    }
}

/** A enrollment. */
public class Enrollment {
    private Long id;
    private int studentId;
    private int courseId;
    private double grade;
    private boolean completed;

    public Enrollment() {
    }

    public Enrollment(Long id, int studentId, int courseId, double grade, boolean completed) {
        this.id = id;
        this.studentId = studentId;
        this.courseId = courseId;
        this.grade = grade;
        this.completed = completed;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getStudentId() {
        return studentId;
    }

    public void setStudentId(int studentId) {
        this.studentId = studentId;
    }

    public int getCourseId() {
        return courseId;
    }

    public void setCourseId(int courseId) {
        this.courseId = courseId;
    }

    public double getGrade() {
        return grade;
    }

    public void setGrade(double grade) {
        this.grade = grade;
    }

    public boolean isCompleted() {
        return completed;
    }

    public void setCompleted(boolean completed) {
        this.completed = completed;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Enrollment other = (Enrollment) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Enrollment{" +
                "id=" + id + ", " +
                "studentId=" + studentId + ", " +
                "courseId=" + courseId + ", " +
                "grade=" + grade + ", " +
                "completed=" + completed +
                "}";
    }
}

/** A room. */
public class Room {
    private Long id;
    private String number;
    private int floor;
    private int beds;
    private double pricePerNight;

    public Room() {
    }

    public Room(Long id, String number, int floor, int beds, double pricePerNight) {
        this.id = id;
        this.number = number;
        this.floor = floor;
        this.beds = beds;
        this.pricePerNight = pricePerNight;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getNumber() {
        return number;
    }

    public void setNumber(String number) {
        this.number = number;
    }

    public int getFloor() {
        return floor;
    }

    public void setFloor(int floor) {
        this.floor = floor;
    }

    public int getBeds() {
        return beds;
    }

    public void setBeds(int beds) {
        this.beds = beds;
    }

    public double getPricePerNight() {
        return pricePerNight;
    }

    public void setPricePerNight(double pricePerNight) {
        this.pricePerNight = pricePerNight;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Room other = (Room) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Room{" +
                "id=" + id + ", " +
                "number=" + number + ", " +
                "floor=" + floor + ", " +
                "beds=" + beds + ", " +
                "pricePerNight=" + pricePerNight +
                "}";
    }
}

/** A booking. */
public class Booking {
    private Long id;
    private int roomId;
    private String guestName;
    private LocalDateTime checkIn;
    private LocalDateTime checkOut;
    private boolean paid;

    public Booking() {
    }

    public Booking(Long id, int roomId, String guestName, LocalDateTime checkIn, LocalDateTime checkOut, boolean paid) {
        this.id = id;
        this.roomId = roomId;
        this.guestName = guestName;
        this.checkIn = checkIn;
        this.checkOut = checkOut;
        this.paid = paid;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getRoomId() {
        return roomId;
    }

    public void setRoomId(int roomId) {
        this.roomId = roomId;
    }

    public String getGuestName() {
        return guestName;
    }

    public void setGuestName(String guestName) {
        this.guestName = guestName;
    }

    public LocalDateTime getCheckIn() {
        return checkIn;
    }

    public void setCheckIn(LocalDateTime checkIn) {
        this.checkIn = checkIn;
    }

    public LocalDateTime getCheckOut() {
        return checkOut;
    }

    public void setCheckOut(LocalDateTime checkOut) {
        this.checkOut = checkOut;
    }

    public boolean isPaid() {
        return paid;
    }

    public void setPaid(boolean paid) {
        this.paid = paid;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Booking other = (Booking) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Booking{" +
                "id=" + id + ", " +
                "roomId=" + roomId + ", " +
                "guestName=" + guestName + ", " +
                "checkIn=" + checkIn + ", " +
                "checkOut=" + checkOut + ", " +
                "paid=" + paid +
                "}";
    }
}

/** A subscription. */
public class Subscription {
    private Long id;
    private int userId;
    private String plan;
    private double price;
    private LocalDateTime renewsAt;
    private boolean isActive;

    public Subscription() {
    }

    public Subscription(Long id, int userId, String plan, double price, LocalDateTime renewsAt, boolean isActive) {
        this.id = id;
        this.userId = userId;
        this.plan = plan;
        this.price = price;
        this.renewsAt = renewsAt;
        this.isActive = isActive;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public String getPlan() {
        return plan;
    }

    public void setPlan(String plan) {
        this.plan = plan;
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    public LocalDateTime getRenewsAt() {
        return renewsAt;
    }

    public void setRenewsAt(LocalDateTime renewsAt) {
        this.renewsAt = renewsAt;
    }

    public boolean isIsActive() {
        return isActive;
    }

    public void setIsActive(boolean isActive) {
        this.isActive = isActive;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Subscription other = (Subscription) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Subscription{" +
                "id=" + id + ", " +
                "userId=" + userId + ", " +
                "plan=" + plan + ", " +
                "price=" + price + ", " +
                "renewsAt=" + renewsAt + ", " +
                "isActive=" + isActive +
                "}";
    }
}

/** A notification. */
public class Notification {
    private Long id;
    private int userId;
    private String title;
    private String message;
    private boolean read;
    private LocalDateTime sentAt;

    public Notification() {
    }

    public Notification(Long id, int userId, String title, String message, boolean read, LocalDateTime sentAt) {
        this.id = id;
        this.userId = userId;
        this.title = title;
        this.message = message;
        this.read = read;
        this.sentAt = sentAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public boolean isRead() {
        return read;
    }

    public void setRead(boolean read) {
        this.read = read;
    }

    public LocalDateTime getSentAt() {
        return sentAt;
    }

    public void setSentAt(LocalDateTime sentAt) {
        this.sentAt = sentAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Notification other = (Notification) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Notification{" +
                "id=" + id + ", " +
                "userId=" + userId + ", " +
                "title=" + title + ", " +
                "message=" + message + ", " +
                "read=" + read + ", " +
                "sentAt=" + sentAt +
                "}";
    }
}

/** A session. */
public class Session {
    private Long id;
    private int userId;
    private String token;
    private LocalDateTime expiresAt;
    private String ipAddress;

    public Session() {
    }

    public Session(Long id, int userId, String token, LocalDateTime expiresAt, String ipAddress) {
        this.id = id;
        this.userId = userId;
        this.token = token;
        this.expiresAt = expiresAt;
        this.ipAddress = ipAddress;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public String getToken() {
        return token;
    }

    public void setToken(String token) {
        this.token = token;
    }

    public LocalDateTime getExpiresAt() {
        return expiresAt;
    }

    public void setExpiresAt(LocalDateTime expiresAt) {
        this.expiresAt = expiresAt;
    }

    public String getIpAddress() {
        return ipAddress;
    }

    public void setIpAddress(String ipAddress) {
        this.ipAddress = ipAddress;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Session other = (Session) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Session{" +
                "id=" + id + ", " +
                "userId=" + userId + ", " +
                "token=" + token + ", " +
                "expiresAt=" + expiresAt + ", " +
                "ipAddress=" + ipAddress +
                "}";
    }
}

/** A audit log. */
public class AuditLog {
    private Long id;
    private int actorId;
    private String action;
    private String target;
    private LocalDateTime createdAt;

    public AuditLog() {
    }

    public AuditLog(Long id, int actorId, String action, String target, LocalDateTime createdAt) {
        this.id = id;
        this.actorId = actorId;
        this.action = action;
        this.target = target;
        this.createdAt = createdAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getActorId() {
        return actorId;
    }

    public void setActorId(int actorId) {
        this.actorId = actorId;
    }

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public String getTarget() {
        return target;
    }

    public void setTarget(String target) {
        this.target = target;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        AuditLog other = (AuditLog) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "AuditLog{" +
                "id=" + id + ", " +
                "actorId=" + actorId + ", " +
                "action=" + action + ", " +
                "target=" + target + ", " +
                "createdAt=" + createdAt +
                "}";
    }
}

/** A setting. */
public class Setting {
    private Long id;
    private String key;
    private String value;
    private String description;

    public Setting() {
    }

    public Setting(Long id, String key, String value, String description) {
        this.id = id;
        this.key = key;
        this.value = value;
        this.description = description;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getKey() {
        return key;
    }

    public void setKey(String key) {
        this.key = key;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Setting other = (Setting) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Setting{" +
                "id=" + id + ", " +
                "key=" + key + ", " +
                "value=" + value + ", " +
                "description=" + description +
                "}";
    }
}

/** A recipe. */
public class Recipe {
    private Long id;
    private String title;
    private int servings;
    private int prepMinutes;
    private int cookMinutes;
    private boolean vegetarian;

    public Recipe() {
    }

    public Recipe(Long id, String title, int servings, int prepMinutes, int cookMinutes, boolean vegetarian) {
        this.id = id;
        this.title = title;
        this.servings = servings;
        this.prepMinutes = prepMinutes;
        this.cookMinutes = cookMinutes;
        this.vegetarian = vegetarian;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public int getServings() {
        return servings;
    }

    public void setServings(int servings) {
        this.servings = servings;
    }

    public int getPrepMinutes() {
        return prepMinutes;
    }

    public void setPrepMinutes(int prepMinutes) {
        this.prepMinutes = prepMinutes;
    }

    public int getCookMinutes() {
        return cookMinutes;
    }

    public void setCookMinutes(int cookMinutes) {
        this.cookMinutes = cookMinutes;
    }

    public boolean isVegetarian() {
        return vegetarian;
    }

    public void setVegetarian(boolean vegetarian) {
        this.vegetarian = vegetarian;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Recipe other = (Recipe) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Recipe{" +
                "id=" + id + ", " +
                "title=" + title + ", " +
                "servings=" + servings + ", " +
                "prepMinutes=" + prepMinutes + ", " +
                "cookMinutes=" + cookMinutes + ", " +
                "vegetarian=" + vegetarian +
                "}";
    }
}

/** A ingredient. */
public class Ingredient {
    private Long id;
    private int recipeId;
    private String name;
    private double amount;
    private String unit;

    public Ingredient() {
    }

    public Ingredient(Long id, int recipeId, String name, double amount, String unit) {
        this.id = id;
        this.recipeId = recipeId;
        this.name = name;
        this.amount = amount;
        this.unit = unit;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public int getRecipeId() {
        return recipeId;
    }

    public void setRecipeId(int recipeId) {
        this.recipeId = recipeId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public String getUnit() {
        return unit;
    }

    public void setUnit(String unit) {
        this.unit = unit;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Ingredient other = (Ingredient) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Ingredient{" +
                "id=" + id + ", " +
                "recipeId=" + recipeId + ", " +
                "name=" + name + ", " +
                "amount=" + amount + ", " +
                "unit=" + unit +
                "}";
    }
}

/** A playlist. */
public class Playlist {
    private Long id;
    private String name;
    private int ownerId;
    private boolean isPublic;
    private int trackCount;

    public Playlist() {
    }

    public Playlist(Long id, String name, int ownerId, boolean isPublic, int trackCount) {
        this.id = id;
        this.name = name;
        this.ownerId = ownerId;
        this.isPublic = isPublic;
        this.trackCount = trackCount;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getOwnerId() {
        return ownerId;
    }

    public void setOwnerId(int ownerId) {
        this.ownerId = ownerId;
    }

    public boolean isIsPublic() {
        return isPublic;
    }

    public void setIsPublic(boolean isPublic) {
        this.isPublic = isPublic;
    }

    public int getTrackCount() {
        return trackCount;
    }

    public void setTrackCount(int trackCount) {
        this.trackCount = trackCount;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Playlist other = (Playlist) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Playlist{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "ownerId=" + ownerId + ", " +
                "isPublic=" + isPublic + ", " +
                "trackCount=" + trackCount +
                "}";
    }
}

/** A track. */
public class Track {
    private Long id;
    private String title;
    private String artist;
    private String album;
    private int durationSeconds;

    public Track() {
    }

    public Track(Long id, String title, String artist, String album, int durationSeconds) {
        this.id = id;
        this.title = title;
        this.artist = artist;
        this.album = album;
        this.durationSeconds = durationSeconds;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getArtist() {
        return artist;
    }

    public void setArtist(String artist) {
        this.artist = artist;
    }

    public String getAlbum() {
        return album;
    }

    public void setAlbum(String album) {
        this.album = album;
    }

    public int getDurationSeconds() {
        return durationSeconds;
    }

    public void setDurationSeconds(int durationSeconds) {
        this.durationSeconds = durationSeconds;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Track other = (Track) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Track{" +
                "id=" + id + ", " +
                "title=" + title + ", " +
                "artist=" + artist + ", " +
                "album=" + album + ", " +
                "durationSeconds=" + durationSeconds +
                "}";
    }
}

/** A device. */
public class Device {
    private Long id;
    private String name;
    private String serialNumber;
    private String firmware;
    private boolean online;
    private LocalDateTime lastSeen;

    public Device() {
    }

    public Device(Long id, String name, String serialNumber, String firmware, boolean online, LocalDateTime lastSeen) {
        this.id = id;
        this.name = name;
        this.serialNumber = serialNumber;
        this.firmware = firmware;
        this.online = online;
        this.lastSeen = lastSeen;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSerialNumber() {
        return serialNumber;
    }

    public void setSerialNumber(String serialNumber) {
        this.serialNumber = serialNumber;
    }

    public String getFirmware() {
        return firmware;
    }

    public void setFirmware(String firmware) {
        this.firmware = firmware;
    }

    public boolean isOnline() {
        return online;
    }

    public void setOnline(boolean online) {
        this.online = online;
    }

    public LocalDateTime getLastSeen() {
        return lastSeen;
    }

    public void setLastSeen(LocalDateTime lastSeen) {
        this.lastSeen = lastSeen;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Device other = (Device) o;
        return Objects.equals(id, other.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Device{" +
                "id=" + id + ", " +
                "name=" + name + ", " +
                "serialNumber=" + serialNumber + ", " +
                "firmware=" + firmware + ", " +
                "online=" + online + ", " +
                "lastSeen=" + lastSeen +
                "}";
    }
}
