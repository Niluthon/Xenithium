package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// writeJSON writes v as a JSON response with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// parseID reads the id path value of the request.
func parseID(r *http.Request) (int64, error) {
	return strconv.ParseInt(r.PathValue("id"), 10, 64)
}

var _ = time.Now

// User is a user.
type User struct {
	ID int64 `json:"id"`
	Username string `json:"username"`
	Email string `json:"email"`
	FullName string `json:"full_name"`
	IsActive bool `json:"is_active"`
	CreatedAt time.Time `json:"created_at"`
}

// UserStore keeps users in memory.
type UserStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]User
}

// NewUserStore returns an empty store.
func NewUserStore() *UserStore {
	return &UserStore{nextID: 1, items: make(map[int64]User)}
}

// Register adds the users routes to mux.
func (s *UserStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /users", s.list)
	mux.HandleFunc("POST /users", s.create)
	mux.HandleFunc("GET /users/{id}", s.get)
	mux.HandleFunc("DELETE /users/{id}", s.delete)
}

func (s *UserStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]User, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *UserStore) create(w http.ResponseWriter, r *http.Request) {
	var user User
	if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	user.ID = s.nextID
	s.nextID++
	s.items[user.ID] = user
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, user)
}

func (s *UserStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	user, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "user not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, user)
}

func (s *UserStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "user not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Product is a product.
type Product struct {
	ID int64 `json:"id"`
	Sku string `json:"sku"`
	Name string `json:"name"`
	Description string `json:"description"`
	Price float64 `json:"price"`
	Stock int64 `json:"stock"`
}

// ProductStore keeps products in memory.
type ProductStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Product
}

// NewProductStore returns an empty store.
func NewProductStore() *ProductStore {
	return &ProductStore{nextID: 1, items: make(map[int64]Product)}
}

// Register adds the products routes to mux.
func (s *ProductStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /products", s.list)
	mux.HandleFunc("POST /products", s.create)
	mux.HandleFunc("GET /products/{id}", s.get)
	mux.HandleFunc("DELETE /products/{id}", s.delete)
}

func (s *ProductStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Product, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *ProductStore) create(w http.ResponseWriter, r *http.Request) {
	var product Product
	if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	product.ID = s.nextID
	s.nextID++
	s.items[product.ID] = product
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, product)
}

func (s *ProductStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	product, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "product not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, product)
}

func (s *ProductStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "product not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Order is a order.
type Order struct {
	ID int64 `json:"id"`
	CustomerId int64 `json:"customer_id"`
	Status string `json:"status"`
	Total float64 `json:"total"`
	Currency string `json:"currency"`
	PlacedAt time.Time `json:"placed_at"`
}

// OrderStore keeps orders in memory.
type OrderStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Order
}

// NewOrderStore returns an empty store.
func NewOrderStore() *OrderStore {
	return &OrderStore{nextID: 1, items: make(map[int64]Order)}
}

// Register adds the orders routes to mux.
func (s *OrderStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /orders", s.list)
	mux.HandleFunc("POST /orders", s.create)
	mux.HandleFunc("GET /orders/{id}", s.get)
	mux.HandleFunc("DELETE /orders/{id}", s.delete)
}

func (s *OrderStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Order, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *OrderStore) create(w http.ResponseWriter, r *http.Request) {
	var order Order
	if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	order.ID = s.nextID
	s.nextID++
	s.items[order.ID] = order
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, order)
}

func (s *OrderStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	order, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "order not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, order)
}

func (s *OrderStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "order not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// OrderItem is a order item.
type OrderItem struct {
	ID int64 `json:"id"`
	OrderId int64 `json:"order_id"`
	ProductId int64 `json:"product_id"`
	Quantity int64 `json:"quantity"`
	UnitPrice float64 `json:"unit_price"`
}

// OrderItemStore keeps order items in memory.
type OrderItemStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]OrderItem
}

// NewOrderItemStore returns an empty store.
func NewOrderItemStore() *OrderItemStore {
	return &OrderItemStore{nextID: 1, items: make(map[int64]OrderItem)}
}

// Register adds the order-items routes to mux.
func (s *OrderItemStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /order-items", s.list)
	mux.HandleFunc("POST /order-items", s.create)
	mux.HandleFunc("GET /order-items/{id}", s.get)
	mux.HandleFunc("DELETE /order-items/{id}", s.delete)
}

func (s *OrderItemStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]OrderItem, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *OrderItemStore) create(w http.ResponseWriter, r *http.Request) {
	var orderItem OrderItem
	if err := json.NewDecoder(r.Body).Decode(&orderItem); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	orderItem.ID = s.nextID
	s.nextID++
	s.items[orderItem.ID] = orderItem
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, orderItem)
}

func (s *OrderItemStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	orderItem, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "order item not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, orderItem)
}

func (s *OrderItemStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "order item not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Customer is a customer.
type Customer struct {
	ID int64 `json:"id"`
	FirstName string `json:"first_name"`
	LastName string `json:"last_name"`
	Email string `json:"email"`
	Phone string `json:"phone"`
	Country string `json:"country"`
}

// CustomerStore keeps customers in memory.
type CustomerStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Customer
}

// NewCustomerStore returns an empty store.
func NewCustomerStore() *CustomerStore {
	return &CustomerStore{nextID: 1, items: make(map[int64]Customer)}
}

// Register adds the customers routes to mux.
func (s *CustomerStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /customers", s.list)
	mux.HandleFunc("POST /customers", s.create)
	mux.HandleFunc("GET /customers/{id}", s.get)
	mux.HandleFunc("DELETE /customers/{id}", s.delete)
}

func (s *CustomerStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Customer, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *CustomerStore) create(w http.ResponseWriter, r *http.Request) {
	var customer Customer
	if err := json.NewDecoder(r.Body).Decode(&customer); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	customer.ID = s.nextID
	s.nextID++
	s.items[customer.ID] = customer
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, customer)
}

func (s *CustomerStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	customer, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "customer not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, customer)
}

func (s *CustomerStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "customer not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Invoice is a invoice.
type Invoice struct {
	ID int64 `json:"id"`
	OrderId int64 `json:"order_id"`
	Number string `json:"number"`
	Amount float64 `json:"amount"`
	DueDate time.Time `json:"due_date"`
	Paid bool `json:"paid"`
}

// InvoiceStore keeps invoices in memory.
type InvoiceStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Invoice
}

// NewInvoiceStore returns an empty store.
func NewInvoiceStore() *InvoiceStore {
	return &InvoiceStore{nextID: 1, items: make(map[int64]Invoice)}
}

// Register adds the invoices routes to mux.
func (s *InvoiceStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /invoices", s.list)
	mux.HandleFunc("POST /invoices", s.create)
	mux.HandleFunc("GET /invoices/{id}", s.get)
	mux.HandleFunc("DELETE /invoices/{id}", s.delete)
}

func (s *InvoiceStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Invoice, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *InvoiceStore) create(w http.ResponseWriter, r *http.Request) {
	var invoice Invoice
	if err := json.NewDecoder(r.Body).Decode(&invoice); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	invoice.ID = s.nextID
	s.nextID++
	s.items[invoice.ID] = invoice
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, invoice)
}

func (s *InvoiceStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	invoice, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "invoice not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, invoice)
}

func (s *InvoiceStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "invoice not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Payment is a payment.
type Payment struct {
	ID int64 `json:"id"`
	InvoiceId int64 `json:"invoice_id"`
	Method string `json:"method"`
	Amount float64 `json:"amount"`
	Reference string `json:"reference"`
	ReceivedAt time.Time `json:"received_at"`
}

// PaymentStore keeps payments in memory.
type PaymentStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Payment
}

// NewPaymentStore returns an empty store.
func NewPaymentStore() *PaymentStore {
	return &PaymentStore{nextID: 1, items: make(map[int64]Payment)}
}

// Register adds the payments routes to mux.
func (s *PaymentStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /payments", s.list)
	mux.HandleFunc("POST /payments", s.create)
	mux.HandleFunc("GET /payments/{id}", s.get)
	mux.HandleFunc("DELETE /payments/{id}", s.delete)
}

func (s *PaymentStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Payment, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *PaymentStore) create(w http.ResponseWriter, r *http.Request) {
	var payment Payment
	if err := json.NewDecoder(r.Body).Decode(&payment); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	payment.ID = s.nextID
	s.nextID++
	s.items[payment.ID] = payment
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, payment)
}

func (s *PaymentStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	payment, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "payment not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, payment)
}

func (s *PaymentStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "payment not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Category is a category.
type Category struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	Slug string `json:"slug"`
	ParentId int64 `json:"parent_id"`
	Position int64 `json:"position"`
}

// CategoryStore keeps categories in memory.
type CategoryStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Category
}

// NewCategoryStore returns an empty store.
func NewCategoryStore() *CategoryStore {
	return &CategoryStore{nextID: 1, items: make(map[int64]Category)}
}

// Register adds the categories routes to mux.
func (s *CategoryStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /categories", s.list)
	mux.HandleFunc("POST /categories", s.create)
	mux.HandleFunc("GET /categories/{id}", s.get)
	mux.HandleFunc("DELETE /categories/{id}", s.delete)
}

func (s *CategoryStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Category, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *CategoryStore) create(w http.ResponseWriter, r *http.Request) {
	var category Category
	if err := json.NewDecoder(r.Body).Decode(&category); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	category.ID = s.nextID
	s.nextID++
	s.items[category.ID] = category
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, category)
}

func (s *CategoryStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	category, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "category not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, category)
}

func (s *CategoryStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "category not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Review is a review.
type Review struct {
	ID int64 `json:"id"`
	ProductId int64 `json:"product_id"`
	UserId int64 `json:"user_id"`
	Rating int64 `json:"rating"`
	Title string `json:"title"`
	Body string `json:"body"`
}

// ReviewStore keeps reviews in memory.
type ReviewStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Review
}

// NewReviewStore returns an empty store.
func NewReviewStore() *ReviewStore {
	return &ReviewStore{nextID: 1, items: make(map[int64]Review)}
}

// Register adds the reviews routes to mux.
func (s *ReviewStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /reviews", s.list)
	mux.HandleFunc("POST /reviews", s.create)
	mux.HandleFunc("GET /reviews/{id}", s.get)
	mux.HandleFunc("DELETE /reviews/{id}", s.delete)
}

func (s *ReviewStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Review, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *ReviewStore) create(w http.ResponseWriter, r *http.Request) {
	var review Review
	if err := json.NewDecoder(r.Body).Decode(&review); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	review.ID = s.nextID
	s.nextID++
	s.items[review.ID] = review
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, review)
}

func (s *ReviewStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	review, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "review not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, review)
}

func (s *ReviewStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "review not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Address is a address.
type Address struct {
	ID int64 `json:"id"`
	CustomerId int64 `json:"customer_id"`
	Street string `json:"street"`
	City string `json:"city"`
	PostalCode string `json:"postal_code"`
	Country string `json:"country"`
}

// AddressStore keeps addresses in memory.
type AddressStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Address
}

// NewAddressStore returns an empty store.
func NewAddressStore() *AddressStore {
	return &AddressStore{nextID: 1, items: make(map[int64]Address)}
}

// Register adds the addresses routes to mux.
func (s *AddressStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /addresses", s.list)
	mux.HandleFunc("POST /addresses", s.create)
	mux.HandleFunc("GET /addresses/{id}", s.get)
	mux.HandleFunc("DELETE /addresses/{id}", s.delete)
}

func (s *AddressStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Address, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *AddressStore) create(w http.ResponseWriter, r *http.Request) {
	var address Address
	if err := json.NewDecoder(r.Body).Decode(&address); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	address.ID = s.nextID
	s.nextID++
	s.items[address.ID] = address
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, address)
}

func (s *AddressStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	address, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "address not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, address)
}

func (s *AddressStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "address not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Shipment is a shipment.
type Shipment struct {
	ID int64 `json:"id"`
	OrderId int64 `json:"order_id"`
	Carrier string `json:"carrier"`
	TrackingNumber string `json:"tracking_number"`
	ShippedAt time.Time `json:"shipped_at"`
	Delivered bool `json:"delivered"`
}

// ShipmentStore keeps shipments in memory.
type ShipmentStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Shipment
}

// NewShipmentStore returns an empty store.
func NewShipmentStore() *ShipmentStore {
	return &ShipmentStore{nextID: 1, items: make(map[int64]Shipment)}
}

// Register adds the shipments routes to mux.
func (s *ShipmentStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /shipments", s.list)
	mux.HandleFunc("POST /shipments", s.create)
	mux.HandleFunc("GET /shipments/{id}", s.get)
	mux.HandleFunc("DELETE /shipments/{id}", s.delete)
}

func (s *ShipmentStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Shipment, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *ShipmentStore) create(w http.ResponseWriter, r *http.Request) {
	var shipment Shipment
	if err := json.NewDecoder(r.Body).Decode(&shipment); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	shipment.ID = s.nextID
	s.nextID++
	s.items[shipment.ID] = shipment
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, shipment)
}

func (s *ShipmentStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	shipment, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "shipment not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, shipment)
}

func (s *ShipmentStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "shipment not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Warehouse is a warehouse.
type Warehouse struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	City string `json:"city"`
	Capacity int64 `json:"capacity"`
	IsOpen bool `json:"is_open"`
}

// WarehouseStore keeps warehouses in memory.
type WarehouseStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Warehouse
}

// NewWarehouseStore returns an empty store.
func NewWarehouseStore() *WarehouseStore {
	return &WarehouseStore{nextID: 1, items: make(map[int64]Warehouse)}
}

// Register adds the warehouses routes to mux.
func (s *WarehouseStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /warehouses", s.list)
	mux.HandleFunc("POST /warehouses", s.create)
	mux.HandleFunc("GET /warehouses/{id}", s.get)
	mux.HandleFunc("DELETE /warehouses/{id}", s.delete)
}

func (s *WarehouseStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Warehouse, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *WarehouseStore) create(w http.ResponseWriter, r *http.Request) {
	var warehouse Warehouse
	if err := json.NewDecoder(r.Body).Decode(&warehouse); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	warehouse.ID = s.nextID
	s.nextID++
	s.items[warehouse.ID] = warehouse
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, warehouse)
}

func (s *WarehouseStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	warehouse, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "warehouse not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, warehouse)
}

func (s *WarehouseStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "warehouse not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Supplier is a supplier.
type Supplier struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	ContactEmail string `json:"contact_email"`
	Phone string `json:"phone"`
	Country string `json:"country"`
	Rating int64 `json:"rating"`
}

// SupplierStore keeps suppliers in memory.
type SupplierStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Supplier
}

// NewSupplierStore returns an empty store.
func NewSupplierStore() *SupplierStore {
	return &SupplierStore{nextID: 1, items: make(map[int64]Supplier)}
}

// Register adds the suppliers routes to mux.
func (s *SupplierStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /suppliers", s.list)
	mux.HandleFunc("POST /suppliers", s.create)
	mux.HandleFunc("GET /suppliers/{id}", s.get)
	mux.HandleFunc("DELETE /suppliers/{id}", s.delete)
}

func (s *SupplierStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Supplier, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *SupplierStore) create(w http.ResponseWriter, r *http.Request) {
	var supplier Supplier
	if err := json.NewDecoder(r.Body).Decode(&supplier); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	supplier.ID = s.nextID
	s.nextID++
	s.items[supplier.ID] = supplier
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, supplier)
}

func (s *SupplierStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	supplier, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "supplier not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, supplier)
}

func (s *SupplierStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "supplier not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Coupon is a coupon.
type Coupon struct {
	ID int64 `json:"id"`
	Code string `json:"code"`
	DiscountPercent int64 `json:"discount_percent"`
	ValidUntil time.Time `json:"valid_until"`
	MaxUses int64 `json:"max_uses"`
	IsActive bool `json:"is_active"`
}

// CouponStore keeps coupons in memory.
type CouponStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Coupon
}

// NewCouponStore returns an empty store.
func NewCouponStore() *CouponStore {
	return &CouponStore{nextID: 1, items: make(map[int64]Coupon)}
}

// Register adds the coupons routes to mux.
func (s *CouponStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /coupons", s.list)
	mux.HandleFunc("POST /coupons", s.create)
	mux.HandleFunc("GET /coupons/{id}", s.get)
	mux.HandleFunc("DELETE /coupons/{id}", s.delete)
}

func (s *CouponStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Coupon, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *CouponStore) create(w http.ResponseWriter, r *http.Request) {
	var coupon Coupon
	if err := json.NewDecoder(r.Body).Decode(&coupon); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	coupon.ID = s.nextID
	s.nextID++
	s.items[coupon.ID] = coupon
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, coupon)
}

func (s *CouponStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	coupon, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "coupon not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, coupon)
}

func (s *CouponStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "coupon not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Employee is a employee.
type Employee struct {
	ID int64 `json:"id"`
	FirstName string `json:"first_name"`
	LastName string `json:"last_name"`
	Email string `json:"email"`
	Department string `json:"department"`
	Salary float64 `json:"salary"`
}

// EmployeeStore keeps employees in memory.
type EmployeeStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Employee
}

// NewEmployeeStore returns an empty store.
func NewEmployeeStore() *EmployeeStore {
	return &EmployeeStore{nextID: 1, items: make(map[int64]Employee)}
}

// Register adds the employees routes to mux.
func (s *EmployeeStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /employees", s.list)
	mux.HandleFunc("POST /employees", s.create)
	mux.HandleFunc("GET /employees/{id}", s.get)
	mux.HandleFunc("DELETE /employees/{id}", s.delete)
}

func (s *EmployeeStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Employee, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *EmployeeStore) create(w http.ResponseWriter, r *http.Request) {
	var employee Employee
	if err := json.NewDecoder(r.Body).Decode(&employee); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	employee.ID = s.nextID
	s.nextID++
	s.items[employee.ID] = employee
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, employee)
}

func (s *EmployeeStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	employee, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "employee not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, employee)
}

func (s *EmployeeStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "employee not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Department is a department.
type Department struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	Code string `json:"code"`
	ManagerId int64 `json:"manager_id"`
	Budget float64 `json:"budget"`
}

// DepartmentStore keeps departments in memory.
type DepartmentStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Department
}

// NewDepartmentStore returns an empty store.
func NewDepartmentStore() *DepartmentStore {
	return &DepartmentStore{nextID: 1, items: make(map[int64]Department)}
}

// Register adds the departments routes to mux.
func (s *DepartmentStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /departments", s.list)
	mux.HandleFunc("POST /departments", s.create)
	mux.HandleFunc("GET /departments/{id}", s.get)
	mux.HandleFunc("DELETE /departments/{id}", s.delete)
}

func (s *DepartmentStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Department, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *DepartmentStore) create(w http.ResponseWriter, r *http.Request) {
	var department Department
	if err := json.NewDecoder(r.Body).Decode(&department); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	department.ID = s.nextID
	s.nextID++
	s.items[department.ID] = department
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, department)
}

func (s *DepartmentStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	department, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "department not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, department)
}

func (s *DepartmentStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "department not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Project is a project.
type Project struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	Description string `json:"description"`
	StartDate time.Time `json:"start_date"`
	EndDate time.Time `json:"end_date"`
	Budget float64 `json:"budget"`
}

// ProjectStore keeps projects in memory.
type ProjectStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Project
}

// NewProjectStore returns an empty store.
func NewProjectStore() *ProjectStore {
	return &ProjectStore{nextID: 1, items: make(map[int64]Project)}
}

// Register adds the projects routes to mux.
func (s *ProjectStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /projects", s.list)
	mux.HandleFunc("POST /projects", s.create)
	mux.HandleFunc("GET /projects/{id}", s.get)
	mux.HandleFunc("DELETE /projects/{id}", s.delete)
}

func (s *ProjectStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Project, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *ProjectStore) create(w http.ResponseWriter, r *http.Request) {
	var project Project
	if err := json.NewDecoder(r.Body).Decode(&project); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	project.ID = s.nextID
	s.nextID++
	s.items[project.ID] = project
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, project)
}

func (s *ProjectStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	project, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "project not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, project)
}

func (s *ProjectStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "project not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Task is a task.
type Task struct {
	ID int64 `json:"id"`
	ProjectId int64 `json:"project_id"`
	Title string `json:"title"`
	Status string `json:"status"`
	Priority int64 `json:"priority"`
	DueDate time.Time `json:"due_date"`
}

// TaskStore keeps tasks in memory.
type TaskStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Task
}

// NewTaskStore returns an empty store.
func NewTaskStore() *TaskStore {
	return &TaskStore{nextID: 1, items: make(map[int64]Task)}
}

// Register adds the tasks routes to mux.
func (s *TaskStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /tasks", s.list)
	mux.HandleFunc("POST /tasks", s.create)
	mux.HandleFunc("GET /tasks/{id}", s.get)
	mux.HandleFunc("DELETE /tasks/{id}", s.delete)
}

func (s *TaskStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Task, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *TaskStore) create(w http.ResponseWriter, r *http.Request) {
	var task Task
	if err := json.NewDecoder(r.Body).Decode(&task); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	task.ID = s.nextID
	s.nextID++
	s.items[task.ID] = task
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, task)
}

func (s *TaskStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	task, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "task not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, task)
}

func (s *TaskStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "task not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Comment is a comment.
type Comment struct {
	ID int64 `json:"id"`
	TaskId int64 `json:"task_id"`
	AuthorId int64 `json:"author_id"`
	Body string `json:"body"`
	CreatedAt time.Time `json:"created_at"`
}

// CommentStore keeps comments in memory.
type CommentStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Comment
}

// NewCommentStore returns an empty store.
func NewCommentStore() *CommentStore {
	return &CommentStore{nextID: 1, items: make(map[int64]Comment)}
}

// Register adds the comments routes to mux.
func (s *CommentStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /comments", s.list)
	mux.HandleFunc("POST /comments", s.create)
	mux.HandleFunc("GET /comments/{id}", s.get)
	mux.HandleFunc("DELETE /comments/{id}", s.delete)
}

func (s *CommentStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Comment, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *CommentStore) create(w http.ResponseWriter, r *http.Request) {
	var comment Comment
	if err := json.NewDecoder(r.Body).Decode(&comment); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	comment.ID = s.nextID
	s.nextID++
	s.items[comment.ID] = comment
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, comment)
}

func (s *CommentStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	comment, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "comment not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, comment)
}

func (s *CommentStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "comment not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Tag is a tag.
type Tag struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	Color string `json:"color"`
	UsageCount int64 `json:"usage_count"`
}

// TagStore keeps tags in memory.
type TagStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Tag
}

// NewTagStore returns an empty store.
func NewTagStore() *TagStore {
	return &TagStore{nextID: 1, items: make(map[int64]Tag)}
}

// Register adds the tags routes to mux.
func (s *TagStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /tags", s.list)
	mux.HandleFunc("POST /tags", s.create)
	mux.HandleFunc("GET /tags/{id}", s.get)
	mux.HandleFunc("DELETE /tags/{id}", s.delete)
}

func (s *TagStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Tag, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *TagStore) create(w http.ResponseWriter, r *http.Request) {
	var tag Tag
	if err := json.NewDecoder(r.Body).Decode(&tag); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	tag.ID = s.nextID
	s.nextID++
	s.items[tag.ID] = tag
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, tag)
}

func (s *TagStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	tag, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "tag not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, tag)
}

func (s *TagStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "tag not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Article is a article.
type Article struct {
	ID int64 `json:"id"`
	Title string `json:"title"`
	Slug string `json:"slug"`
	Body string `json:"body"`
	AuthorId int64 `json:"author_id"`
	Published bool `json:"published"`
}

// ArticleStore keeps articles in memory.
type ArticleStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Article
}

// NewArticleStore returns an empty store.
func NewArticleStore() *ArticleStore {
	return &ArticleStore{nextID: 1, items: make(map[int64]Article)}
}

// Register adds the articles routes to mux.
func (s *ArticleStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /articles", s.list)
	mux.HandleFunc("POST /articles", s.create)
	mux.HandleFunc("GET /articles/{id}", s.get)
	mux.HandleFunc("DELETE /articles/{id}", s.delete)
}

func (s *ArticleStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Article, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *ArticleStore) create(w http.ResponseWriter, r *http.Request) {
	var article Article
	if err := json.NewDecoder(r.Body).Decode(&article); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	article.ID = s.nextID
	s.nextID++
	s.items[article.ID] = article
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, article)
}

func (s *ArticleStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	article, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "article not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, article)
}

func (s *ArticleStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "article not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Author is a author.
type Author struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	Bio string `json:"bio"`
	Email string `json:"email"`
	Website string `json:"website"`
}

// AuthorStore keeps authors in memory.
type AuthorStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Author
}

// NewAuthorStore returns an empty store.
func NewAuthorStore() *AuthorStore {
	return &AuthorStore{nextID: 1, items: make(map[int64]Author)}
}

// Register adds the authors routes to mux.
func (s *AuthorStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /authors", s.list)
	mux.HandleFunc("POST /authors", s.create)
	mux.HandleFunc("GET /authors/{id}", s.get)
	mux.HandleFunc("DELETE /authors/{id}", s.delete)
}

func (s *AuthorStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Author, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *AuthorStore) create(w http.ResponseWriter, r *http.Request) {
	var author Author
	if err := json.NewDecoder(r.Body).Decode(&author); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	author.ID = s.nextID
	s.nextID++
	s.items[author.ID] = author
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, author)
}

func (s *AuthorStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	author, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "author not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, author)
}

func (s *AuthorStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "author not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Book is a book.
type Book struct {
	ID int64 `json:"id"`
	Title string `json:"title"`
	Isbn string `json:"isbn"`
	AuthorId int64 `json:"author_id"`
	Year int64 `json:"year"`
	Pages int64 `json:"pages"`
}

// BookStore keeps books in memory.
type BookStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Book
}

// NewBookStore returns an empty store.
func NewBookStore() *BookStore {
	return &BookStore{nextID: 1, items: make(map[int64]Book)}
}

// Register adds the books routes to mux.
func (s *BookStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /books", s.list)
	mux.HandleFunc("POST /books", s.create)
	mux.HandleFunc("GET /books/{id}", s.get)
	mux.HandleFunc("DELETE /books/{id}", s.delete)
}

func (s *BookStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Book, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *BookStore) create(w http.ResponseWriter, r *http.Request) {
	var book Book
	if err := json.NewDecoder(r.Body).Decode(&book); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	book.ID = s.nextID
	s.nextID++
	s.items[book.ID] = book
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, book)
}

func (s *BookStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	book, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "book not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, book)
}

func (s *BookStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "book not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Library is a library.
type Library struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	City string `json:"city"`
	OpeningHour int64 `json:"opening_hour"`
	ClosingHour int64 `json:"closing_hour"`
}

// LibraryStore keeps libraries in memory.
type LibraryStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Library
}

// NewLibraryStore returns an empty store.
func NewLibraryStore() *LibraryStore {
	return &LibraryStore{nextID: 1, items: make(map[int64]Library)}
}

// Register adds the libraries routes to mux.
func (s *LibraryStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /libraries", s.list)
	mux.HandleFunc("POST /libraries", s.create)
	mux.HandleFunc("GET /libraries/{id}", s.get)
	mux.HandleFunc("DELETE /libraries/{id}", s.delete)
}

func (s *LibraryStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Library, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *LibraryStore) create(w http.ResponseWriter, r *http.Request) {
	var library Library
	if err := json.NewDecoder(r.Body).Decode(&library); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	library.ID = s.nextID
	s.nextID++
	s.items[library.ID] = library
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, library)
}

func (s *LibraryStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	library, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "library not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, library)
}

func (s *LibraryStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "library not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Loan is a loan.
type Loan struct {
	ID int64 `json:"id"`
	BookId int64 `json:"book_id"`
	MemberId int64 `json:"member_id"`
	BorrowedAt time.Time `json:"borrowed_at"`
	DueDate time.Time `json:"due_date"`
	Returned bool `json:"returned"`
}

// LoanStore keeps loans in memory.
type LoanStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Loan
}

// NewLoanStore returns an empty store.
func NewLoanStore() *LoanStore {
	return &LoanStore{nextID: 1, items: make(map[int64]Loan)}
}

// Register adds the loans routes to mux.
func (s *LoanStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /loans", s.list)
	mux.HandleFunc("POST /loans", s.create)
	mux.HandleFunc("GET /loans/{id}", s.get)
	mux.HandleFunc("DELETE /loans/{id}", s.delete)
}

func (s *LoanStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Loan, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *LoanStore) create(w http.ResponseWriter, r *http.Request) {
	var loan Loan
	if err := json.NewDecoder(r.Body).Decode(&loan); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	loan.ID = s.nextID
	s.nextID++
	s.items[loan.ID] = loan
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, loan)
}

func (s *LoanStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	loan, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "loan not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, loan)
}

func (s *LoanStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "loan not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Member is a member.
type Member struct {
	ID int64 `json:"id"`
	FirstName string `json:"first_name"`
	LastName string `json:"last_name"`
	Email string `json:"email"`
	JoinedAt time.Time `json:"joined_at"`
}

// MemberStore keeps members in memory.
type MemberStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Member
}

// NewMemberStore returns an empty store.
func NewMemberStore() *MemberStore {
	return &MemberStore{nextID: 1, items: make(map[int64]Member)}
}

// Register adds the members routes to mux.
func (s *MemberStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /members", s.list)
	mux.HandleFunc("POST /members", s.create)
	mux.HandleFunc("GET /members/{id}", s.get)
	mux.HandleFunc("DELETE /members/{id}", s.delete)
}

func (s *MemberStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Member, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *MemberStore) create(w http.ResponseWriter, r *http.Request) {
	var member Member
	if err := json.NewDecoder(r.Body).Decode(&member); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	member.ID = s.nextID
	s.nextID++
	s.items[member.ID] = member
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, member)
}

func (s *MemberStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	member, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "member not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, member)
}

func (s *MemberStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "member not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Event is a event.
type Event struct {
	ID int64 `json:"id"`
	Title string `json:"title"`
	Location string `json:"location"`
	StartsAt time.Time `json:"starts_at"`
	EndsAt time.Time `json:"ends_at"`
	Capacity int64 `json:"capacity"`
}

// EventStore keeps events in memory.
type EventStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Event
}

// NewEventStore returns an empty store.
func NewEventStore() *EventStore {
	return &EventStore{nextID: 1, items: make(map[int64]Event)}
}

// Register adds the events routes to mux.
func (s *EventStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /events", s.list)
	mux.HandleFunc("POST /events", s.create)
	mux.HandleFunc("GET /events/{id}", s.get)
	mux.HandleFunc("DELETE /events/{id}", s.delete)
}

func (s *EventStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Event, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *EventStore) create(w http.ResponseWriter, r *http.Request) {
	var event Event
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	event.ID = s.nextID
	s.nextID++
	s.items[event.ID] = event
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, event)
}

func (s *EventStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	event, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "event not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, event)
}

func (s *EventStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "event not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Ticket is a ticket.
type Ticket struct {
	ID int64 `json:"id"`
	EventId int64 `json:"event_id"`
	HolderName string `json:"holder_name"`
	Seat string `json:"seat"`
	Price float64 `json:"price"`
	CheckedIn bool `json:"checked_in"`
}

// TicketStore keeps tickets in memory.
type TicketStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Ticket
}

// NewTicketStore returns an empty store.
func NewTicketStore() *TicketStore {
	return &TicketStore{nextID: 1, items: make(map[int64]Ticket)}
}

// Register adds the tickets routes to mux.
func (s *TicketStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /tickets", s.list)
	mux.HandleFunc("POST /tickets", s.create)
	mux.HandleFunc("GET /tickets/{id}", s.get)
	mux.HandleFunc("DELETE /tickets/{id}", s.delete)
}

func (s *TicketStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Ticket, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *TicketStore) create(w http.ResponseWriter, r *http.Request) {
	var ticket Ticket
	if err := json.NewDecoder(r.Body).Decode(&ticket); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	ticket.ID = s.nextID
	s.nextID++
	s.items[ticket.ID] = ticket
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, ticket)
}

func (s *TicketStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	ticket, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "ticket not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, ticket)
}

func (s *TicketStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "ticket not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Venue is a venue.
type Venue struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	Address string `json:"address"`
	City string `json:"city"`
	Capacity int64 `json:"capacity"`
}

// VenueStore keeps venues in memory.
type VenueStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Venue
}

// NewVenueStore returns an empty store.
func NewVenueStore() *VenueStore {
	return &VenueStore{nextID: 1, items: make(map[int64]Venue)}
}

// Register adds the venues routes to mux.
func (s *VenueStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /venues", s.list)
	mux.HandleFunc("POST /venues", s.create)
	mux.HandleFunc("GET /venues/{id}", s.get)
	mux.HandleFunc("DELETE /venues/{id}", s.delete)
}

func (s *VenueStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Venue, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *VenueStore) create(w http.ResponseWriter, r *http.Request) {
	var venue Venue
	if err := json.NewDecoder(r.Body).Decode(&venue); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	venue.ID = s.nextID
	s.nextID++
	s.items[venue.ID] = venue
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, venue)
}

func (s *VenueStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	venue, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "venue not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, venue)
}

func (s *VenueStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "venue not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Vehicle is a vehicle.
type Vehicle struct {
	ID int64 `json:"id"`
	Plate string `json:"plate"`
	Make string `json:"make"`
	Model string `json:"model"`
	Year int64 `json:"year"`
	Mileage int64 `json:"mileage"`
}

// VehicleStore keeps vehicles in memory.
type VehicleStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Vehicle
}

// NewVehicleStore returns an empty store.
func NewVehicleStore() *VehicleStore {
	return &VehicleStore{nextID: 1, items: make(map[int64]Vehicle)}
}

// Register adds the vehicles routes to mux.
func (s *VehicleStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /vehicles", s.list)
	mux.HandleFunc("POST /vehicles", s.create)
	mux.HandleFunc("GET /vehicles/{id}", s.get)
	mux.HandleFunc("DELETE /vehicles/{id}", s.delete)
}

func (s *VehicleStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Vehicle, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *VehicleStore) create(w http.ResponseWriter, r *http.Request) {
	var vehicle Vehicle
	if err := json.NewDecoder(r.Body).Decode(&vehicle); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	vehicle.ID = s.nextID
	s.nextID++
	s.items[vehicle.ID] = vehicle
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, vehicle)
}

func (s *VehicleStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	vehicle, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "vehicle not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, vehicle)
}

func (s *VehicleStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "vehicle not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Driver is a driver.
type Driver struct {
	ID int64 `json:"id"`
	FirstName string `json:"first_name"`
	LastName string `json:"last_name"`
	LicenseNumber string `json:"license_number"`
	Rating float64 `json:"rating"`
}

// DriverStore keeps drivers in memory.
type DriverStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Driver
}

// NewDriverStore returns an empty store.
func NewDriverStore() *DriverStore {
	return &DriverStore{nextID: 1, items: make(map[int64]Driver)}
}

// Register adds the drivers routes to mux.
func (s *DriverStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /drivers", s.list)
	mux.HandleFunc("POST /drivers", s.create)
	mux.HandleFunc("GET /drivers/{id}", s.get)
	mux.HandleFunc("DELETE /drivers/{id}", s.delete)
}

func (s *DriverStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Driver, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *DriverStore) create(w http.ResponseWriter, r *http.Request) {
	var driver Driver
	if err := json.NewDecoder(r.Body).Decode(&driver); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	driver.ID = s.nextID
	s.nextID++
	s.items[driver.ID] = driver
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, driver)
}

func (s *DriverStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	driver, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "driver not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, driver)
}

func (s *DriverStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "driver not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Trip is a trip.
type Trip struct {
	ID int64 `json:"id"`
	VehicleId int64 `json:"vehicle_id"`
	DriverId int64 `json:"driver_id"`
	DistanceKm float64 `json:"distance_km"`
	StartedAt time.Time `json:"started_at"`
	Fare float64 `json:"fare"`
}

// TripStore keeps trips in memory.
type TripStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Trip
}

// NewTripStore returns an empty store.
func NewTripStore() *TripStore {
	return &TripStore{nextID: 1, items: make(map[int64]Trip)}
}

// Register adds the trips routes to mux.
func (s *TripStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /trips", s.list)
	mux.HandleFunc("POST /trips", s.create)
	mux.HandleFunc("GET /trips/{id}", s.get)
	mux.HandleFunc("DELETE /trips/{id}", s.delete)
}

func (s *TripStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Trip, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *TripStore) create(w http.ResponseWriter, r *http.Request) {
	var trip Trip
	if err := json.NewDecoder(r.Body).Decode(&trip); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	trip.ID = s.nextID
	s.nextID++
	s.items[trip.ID] = trip
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, trip)
}

func (s *TripStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	trip, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "trip not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, trip)
}

func (s *TripStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "trip not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Patient is a patient.
type Patient struct {
	ID int64 `json:"id"`
	FirstName string `json:"first_name"`
	LastName string `json:"last_name"`
	BirthDate time.Time `json:"birth_date"`
	Phone string `json:"phone"`
}

// PatientStore keeps patients in memory.
type PatientStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Patient
}

// NewPatientStore returns an empty store.
func NewPatientStore() *PatientStore {
	return &PatientStore{nextID: 1, items: make(map[int64]Patient)}
}

// Register adds the patients routes to mux.
func (s *PatientStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /patients", s.list)
	mux.HandleFunc("POST /patients", s.create)
	mux.HandleFunc("GET /patients/{id}", s.get)
	mux.HandleFunc("DELETE /patients/{id}", s.delete)
}

func (s *PatientStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Patient, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *PatientStore) create(w http.ResponseWriter, r *http.Request) {
	var patient Patient
	if err := json.NewDecoder(r.Body).Decode(&patient); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	patient.ID = s.nextID
	s.nextID++
	s.items[patient.ID] = patient
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, patient)
}

func (s *PatientStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	patient, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "patient not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, patient)
}

func (s *PatientStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "patient not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Doctor is a doctor.
type Doctor struct {
	ID int64 `json:"id"`
	FirstName string `json:"first_name"`
	LastName string `json:"last_name"`
	Specialty string `json:"specialty"`
	Room string `json:"room"`
}

// DoctorStore keeps doctors in memory.
type DoctorStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Doctor
}

// NewDoctorStore returns an empty store.
func NewDoctorStore() *DoctorStore {
	return &DoctorStore{nextID: 1, items: make(map[int64]Doctor)}
}

// Register adds the doctors routes to mux.
func (s *DoctorStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /doctors", s.list)
	mux.HandleFunc("POST /doctors", s.create)
	mux.HandleFunc("GET /doctors/{id}", s.get)
	mux.HandleFunc("DELETE /doctors/{id}", s.delete)
}

func (s *DoctorStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Doctor, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *DoctorStore) create(w http.ResponseWriter, r *http.Request) {
	var doctor Doctor
	if err := json.NewDecoder(r.Body).Decode(&doctor); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	doctor.ID = s.nextID
	s.nextID++
	s.items[doctor.ID] = doctor
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, doctor)
}

func (s *DoctorStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	doctor, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "doctor not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, doctor)
}

func (s *DoctorStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "doctor not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Appointment is a appointment.
type Appointment struct {
	ID int64 `json:"id"`
	PatientId int64 `json:"patient_id"`
	DoctorId int64 `json:"doctor_id"`
	ScheduledAt time.Time `json:"scheduled_at"`
	Reason string `json:"reason"`
	Confirmed bool `json:"confirmed"`
}

// AppointmentStore keeps appointments in memory.
type AppointmentStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Appointment
}

// NewAppointmentStore returns an empty store.
func NewAppointmentStore() *AppointmentStore {
	return &AppointmentStore{nextID: 1, items: make(map[int64]Appointment)}
}

// Register adds the appointments routes to mux.
func (s *AppointmentStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /appointments", s.list)
	mux.HandleFunc("POST /appointments", s.create)
	mux.HandleFunc("GET /appointments/{id}", s.get)
	mux.HandleFunc("DELETE /appointments/{id}", s.delete)
}

func (s *AppointmentStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Appointment, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *AppointmentStore) create(w http.ResponseWriter, r *http.Request) {
	var appointment Appointment
	if err := json.NewDecoder(r.Body).Decode(&appointment); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	appointment.ID = s.nextID
	s.nextID++
	s.items[appointment.ID] = appointment
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, appointment)
}

func (s *AppointmentStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	appointment, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "appointment not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, appointment)
}

func (s *AppointmentStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "appointment not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Course is a course.
type Course struct {
	ID int64 `json:"id"`
	Title string `json:"title"`
	Code string `json:"code"`
	Credits int64 `json:"credits"`
	TeacherId int64 `json:"teacher_id"`
}

// CourseStore keeps courses in memory.
type CourseStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Course
}

// NewCourseStore returns an empty store.
func NewCourseStore() *CourseStore {
	return &CourseStore{nextID: 1, items: make(map[int64]Course)}
}

// Register adds the courses routes to mux.
func (s *CourseStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /courses", s.list)
	mux.HandleFunc("POST /courses", s.create)
	mux.HandleFunc("GET /courses/{id}", s.get)
	mux.HandleFunc("DELETE /courses/{id}", s.delete)
}

func (s *CourseStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Course, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *CourseStore) create(w http.ResponseWriter, r *http.Request) {
	var course Course
	if err := json.NewDecoder(r.Body).Decode(&course); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	course.ID = s.nextID
	s.nextID++
	s.items[course.ID] = course
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, course)
}

func (s *CourseStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	course, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "course not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, course)
}

func (s *CourseStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "course not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Student is a student.
type Student struct {
	ID int64 `json:"id"`
	FirstName string `json:"first_name"`
	LastName string `json:"last_name"`
	Email string `json:"email"`
	EnrolledAt time.Time `json:"enrolled_at"`
}

// StudentStore keeps students in memory.
type StudentStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Student
}

// NewStudentStore returns an empty store.
func NewStudentStore() *StudentStore {
	return &StudentStore{nextID: 1, items: make(map[int64]Student)}
}

// Register adds the students routes to mux.
func (s *StudentStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /students", s.list)
	mux.HandleFunc("POST /students", s.create)
	mux.HandleFunc("GET /students/{id}", s.get)
	mux.HandleFunc("DELETE /students/{id}", s.delete)
}

func (s *StudentStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Student, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *StudentStore) create(w http.ResponseWriter, r *http.Request) {
	var student Student
	if err := json.NewDecoder(r.Body).Decode(&student); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	student.ID = s.nextID
	s.nextID++
	s.items[student.ID] = student
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, student)
}

func (s *StudentStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	student, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "student not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, student)
}

func (s *StudentStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "student not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Enrollment is a enrollment.
type Enrollment struct {
	ID int64 `json:"id"`
	StudentId int64 `json:"student_id"`
	CourseId int64 `json:"course_id"`
	Grade float64 `json:"grade"`
	Completed bool `json:"completed"`
}

// EnrollmentStore keeps enrollments in memory.
type EnrollmentStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Enrollment
}

// NewEnrollmentStore returns an empty store.
func NewEnrollmentStore() *EnrollmentStore {
	return &EnrollmentStore{nextID: 1, items: make(map[int64]Enrollment)}
}

// Register adds the enrollments routes to mux.
func (s *EnrollmentStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /enrollments", s.list)
	mux.HandleFunc("POST /enrollments", s.create)
	mux.HandleFunc("GET /enrollments/{id}", s.get)
	mux.HandleFunc("DELETE /enrollments/{id}", s.delete)
}

func (s *EnrollmentStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Enrollment, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *EnrollmentStore) create(w http.ResponseWriter, r *http.Request) {
	var enrollment Enrollment
	if err := json.NewDecoder(r.Body).Decode(&enrollment); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	enrollment.ID = s.nextID
	s.nextID++
	s.items[enrollment.ID] = enrollment
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, enrollment)
}

func (s *EnrollmentStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	enrollment, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "enrollment not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, enrollment)
}

func (s *EnrollmentStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "enrollment not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Room is a room.
type Room struct {
	ID int64 `json:"id"`
	Number string `json:"number"`
	Floor int64 `json:"floor"`
	Beds int64 `json:"beds"`
	PricePerNight float64 `json:"price_per_night"`
}

// RoomStore keeps rooms in memory.
type RoomStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Room
}

// NewRoomStore returns an empty store.
func NewRoomStore() *RoomStore {
	return &RoomStore{nextID: 1, items: make(map[int64]Room)}
}

// Register adds the rooms routes to mux.
func (s *RoomStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /rooms", s.list)
	mux.HandleFunc("POST /rooms", s.create)
	mux.HandleFunc("GET /rooms/{id}", s.get)
	mux.HandleFunc("DELETE /rooms/{id}", s.delete)
}

func (s *RoomStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Room, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *RoomStore) create(w http.ResponseWriter, r *http.Request) {
	var room Room
	if err := json.NewDecoder(r.Body).Decode(&room); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	room.ID = s.nextID
	s.nextID++
	s.items[room.ID] = room
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, room)
}

func (s *RoomStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	room, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "room not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, room)
}

func (s *RoomStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "room not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Booking is a booking.
type Booking struct {
	ID int64 `json:"id"`
	RoomId int64 `json:"room_id"`
	GuestName string `json:"guest_name"`
	CheckIn time.Time `json:"check_in"`
	CheckOut time.Time `json:"check_out"`
	Paid bool `json:"paid"`
}

// BookingStore keeps bookings in memory.
type BookingStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Booking
}

// NewBookingStore returns an empty store.
func NewBookingStore() *BookingStore {
	return &BookingStore{nextID: 1, items: make(map[int64]Booking)}
}

// Register adds the bookings routes to mux.
func (s *BookingStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /bookings", s.list)
	mux.HandleFunc("POST /bookings", s.create)
	mux.HandleFunc("GET /bookings/{id}", s.get)
	mux.HandleFunc("DELETE /bookings/{id}", s.delete)
}

func (s *BookingStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Booking, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *BookingStore) create(w http.ResponseWriter, r *http.Request) {
	var booking Booking
	if err := json.NewDecoder(r.Body).Decode(&booking); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	booking.ID = s.nextID
	s.nextID++
	s.items[booking.ID] = booking
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, booking)
}

func (s *BookingStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	booking, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "booking not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, booking)
}

func (s *BookingStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "booking not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Subscription is a subscription.
type Subscription struct {
	ID int64 `json:"id"`
	UserId int64 `json:"user_id"`
	Plan string `json:"plan"`
	Price float64 `json:"price"`
	RenewsAt time.Time `json:"renews_at"`
	IsActive bool `json:"is_active"`
}

// SubscriptionStore keeps subscriptions in memory.
type SubscriptionStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Subscription
}

// NewSubscriptionStore returns an empty store.
func NewSubscriptionStore() *SubscriptionStore {
	return &SubscriptionStore{nextID: 1, items: make(map[int64]Subscription)}
}

// Register adds the subscriptions routes to mux.
func (s *SubscriptionStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /subscriptions", s.list)
	mux.HandleFunc("POST /subscriptions", s.create)
	mux.HandleFunc("GET /subscriptions/{id}", s.get)
	mux.HandleFunc("DELETE /subscriptions/{id}", s.delete)
}

func (s *SubscriptionStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Subscription, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *SubscriptionStore) create(w http.ResponseWriter, r *http.Request) {
	var subscription Subscription
	if err := json.NewDecoder(r.Body).Decode(&subscription); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	subscription.ID = s.nextID
	s.nextID++
	s.items[subscription.ID] = subscription
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, subscription)
}

func (s *SubscriptionStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	subscription, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "subscription not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, subscription)
}

func (s *SubscriptionStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "subscription not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Notification is a notification.
type Notification struct {
	ID int64 `json:"id"`
	UserId int64 `json:"user_id"`
	Title string `json:"title"`
	Message string `json:"message"`
	Read bool `json:"read"`
	SentAt time.Time `json:"sent_at"`
}

// NotificationStore keeps notifications in memory.
type NotificationStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Notification
}

// NewNotificationStore returns an empty store.
func NewNotificationStore() *NotificationStore {
	return &NotificationStore{nextID: 1, items: make(map[int64]Notification)}
}

// Register adds the notifications routes to mux.
func (s *NotificationStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /notifications", s.list)
	mux.HandleFunc("POST /notifications", s.create)
	mux.HandleFunc("GET /notifications/{id}", s.get)
	mux.HandleFunc("DELETE /notifications/{id}", s.delete)
}

func (s *NotificationStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Notification, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *NotificationStore) create(w http.ResponseWriter, r *http.Request) {
	var notification Notification
	if err := json.NewDecoder(r.Body).Decode(&notification); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	notification.ID = s.nextID
	s.nextID++
	s.items[notification.ID] = notification
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, notification)
}

func (s *NotificationStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	notification, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "notification not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, notification)
}

func (s *NotificationStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "notification not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Session is a session.
type Session struct {
	ID int64 `json:"id"`
	UserId int64 `json:"user_id"`
	Token string `json:"token"`
	ExpiresAt time.Time `json:"expires_at"`
	IpAddress string `json:"ip_address"`
}

// SessionStore keeps sessions in memory.
type SessionStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Session
}

// NewSessionStore returns an empty store.
func NewSessionStore() *SessionStore {
	return &SessionStore{nextID: 1, items: make(map[int64]Session)}
}

// Register adds the sessions routes to mux.
func (s *SessionStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /sessions", s.list)
	mux.HandleFunc("POST /sessions", s.create)
	mux.HandleFunc("GET /sessions/{id}", s.get)
	mux.HandleFunc("DELETE /sessions/{id}", s.delete)
}

func (s *SessionStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Session, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *SessionStore) create(w http.ResponseWriter, r *http.Request) {
	var session Session
	if err := json.NewDecoder(r.Body).Decode(&session); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	session.ID = s.nextID
	s.nextID++
	s.items[session.ID] = session
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, session)
}

func (s *SessionStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	session, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, session)
}

func (s *SessionStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// AuditLog is a audit log.
type AuditLog struct {
	ID int64 `json:"id"`
	ActorId int64 `json:"actor_id"`
	Action string `json:"action"`
	Target string `json:"target"`
	CreatedAt time.Time `json:"created_at"`
}

// AuditLogStore keeps audit logs in memory.
type AuditLogStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]AuditLog
}

// NewAuditLogStore returns an empty store.
func NewAuditLogStore() *AuditLogStore {
	return &AuditLogStore{nextID: 1, items: make(map[int64]AuditLog)}
}

// Register adds the audit-logs routes to mux.
func (s *AuditLogStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /audit-logs", s.list)
	mux.HandleFunc("POST /audit-logs", s.create)
	mux.HandleFunc("GET /audit-logs/{id}", s.get)
	mux.HandleFunc("DELETE /audit-logs/{id}", s.delete)
}

func (s *AuditLogStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]AuditLog, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *AuditLogStore) create(w http.ResponseWriter, r *http.Request) {
	var auditLog AuditLog
	if err := json.NewDecoder(r.Body).Decode(&auditLog); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	auditLog.ID = s.nextID
	s.nextID++
	s.items[auditLog.ID] = auditLog
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, auditLog)
}

func (s *AuditLogStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	auditLog, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "audit log not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, auditLog)
}

func (s *AuditLogStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "audit log not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Setting is a setting.
type Setting struct {
	ID int64 `json:"id"`
	Key string `json:"key"`
	Value string `json:"value"`
	Description string `json:"description"`
}

// SettingStore keeps settings in memory.
type SettingStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Setting
}

// NewSettingStore returns an empty store.
func NewSettingStore() *SettingStore {
	return &SettingStore{nextID: 1, items: make(map[int64]Setting)}
}

// Register adds the settings routes to mux.
func (s *SettingStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /settings", s.list)
	mux.HandleFunc("POST /settings", s.create)
	mux.HandleFunc("GET /settings/{id}", s.get)
	mux.HandleFunc("DELETE /settings/{id}", s.delete)
}

func (s *SettingStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Setting, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *SettingStore) create(w http.ResponseWriter, r *http.Request) {
	var setting Setting
	if err := json.NewDecoder(r.Body).Decode(&setting); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	setting.ID = s.nextID
	s.nextID++
	s.items[setting.ID] = setting
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, setting)
}

func (s *SettingStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	setting, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "setting not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, setting)
}

func (s *SettingStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "setting not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Recipe is a recipe.
type Recipe struct {
	ID int64 `json:"id"`
	Title string `json:"title"`
	Servings int64 `json:"servings"`
	PrepMinutes int64 `json:"prep_minutes"`
	CookMinutes int64 `json:"cook_minutes"`
	Vegetarian bool `json:"vegetarian"`
}

// RecipeStore keeps recipes in memory.
type RecipeStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Recipe
}

// NewRecipeStore returns an empty store.
func NewRecipeStore() *RecipeStore {
	return &RecipeStore{nextID: 1, items: make(map[int64]Recipe)}
}

// Register adds the recipes routes to mux.
func (s *RecipeStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /recipes", s.list)
	mux.HandleFunc("POST /recipes", s.create)
	mux.HandleFunc("GET /recipes/{id}", s.get)
	mux.HandleFunc("DELETE /recipes/{id}", s.delete)
}

func (s *RecipeStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Recipe, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *RecipeStore) create(w http.ResponseWriter, r *http.Request) {
	var recipe Recipe
	if err := json.NewDecoder(r.Body).Decode(&recipe); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	recipe.ID = s.nextID
	s.nextID++
	s.items[recipe.ID] = recipe
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, recipe)
}

func (s *RecipeStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	recipe, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "recipe not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, recipe)
}

func (s *RecipeStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "recipe not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Ingredient is a ingredient.
type Ingredient struct {
	ID int64 `json:"id"`
	RecipeId int64 `json:"recipe_id"`
	Name string `json:"name"`
	Amount float64 `json:"amount"`
	Unit string `json:"unit"`
}

// IngredientStore keeps ingredients in memory.
type IngredientStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Ingredient
}

// NewIngredientStore returns an empty store.
func NewIngredientStore() *IngredientStore {
	return &IngredientStore{nextID: 1, items: make(map[int64]Ingredient)}
}

// Register adds the ingredients routes to mux.
func (s *IngredientStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /ingredients", s.list)
	mux.HandleFunc("POST /ingredients", s.create)
	mux.HandleFunc("GET /ingredients/{id}", s.get)
	mux.HandleFunc("DELETE /ingredients/{id}", s.delete)
}

func (s *IngredientStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Ingredient, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *IngredientStore) create(w http.ResponseWriter, r *http.Request) {
	var ingredient Ingredient
	if err := json.NewDecoder(r.Body).Decode(&ingredient); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	ingredient.ID = s.nextID
	s.nextID++
	s.items[ingredient.ID] = ingredient
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, ingredient)
}

func (s *IngredientStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	ingredient, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "ingredient not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, ingredient)
}

func (s *IngredientStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "ingredient not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Playlist is a playlist.
type Playlist struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	OwnerId int64 `json:"owner_id"`
	IsPublic bool `json:"is_public"`
	TrackCount int64 `json:"track_count"`
}

// PlaylistStore keeps playlists in memory.
type PlaylistStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Playlist
}

// NewPlaylistStore returns an empty store.
func NewPlaylistStore() *PlaylistStore {
	return &PlaylistStore{nextID: 1, items: make(map[int64]Playlist)}
}

// Register adds the playlists routes to mux.
func (s *PlaylistStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /playlists", s.list)
	mux.HandleFunc("POST /playlists", s.create)
	mux.HandleFunc("GET /playlists/{id}", s.get)
	mux.HandleFunc("DELETE /playlists/{id}", s.delete)
}

func (s *PlaylistStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Playlist, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *PlaylistStore) create(w http.ResponseWriter, r *http.Request) {
	var playlist Playlist
	if err := json.NewDecoder(r.Body).Decode(&playlist); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	playlist.ID = s.nextID
	s.nextID++
	s.items[playlist.ID] = playlist
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, playlist)
}

func (s *PlaylistStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	playlist, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "playlist not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, playlist)
}

func (s *PlaylistStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "playlist not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Track is a track.
type Track struct {
	ID int64 `json:"id"`
	Title string `json:"title"`
	Artist string `json:"artist"`
	Album string `json:"album"`
	DurationSeconds int64 `json:"duration_seconds"`
}

// TrackStore keeps tracks in memory.
type TrackStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Track
}

// NewTrackStore returns an empty store.
func NewTrackStore() *TrackStore {
	return &TrackStore{nextID: 1, items: make(map[int64]Track)}
}

// Register adds the tracks routes to mux.
func (s *TrackStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /tracks", s.list)
	mux.HandleFunc("POST /tracks", s.create)
	mux.HandleFunc("GET /tracks/{id}", s.get)
	mux.HandleFunc("DELETE /tracks/{id}", s.delete)
}

func (s *TrackStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Track, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *TrackStore) create(w http.ResponseWriter, r *http.Request) {
	var track Track
	if err := json.NewDecoder(r.Body).Decode(&track); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	track.ID = s.nextID
	s.nextID++
	s.items[track.ID] = track
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, track)
}

func (s *TrackStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	track, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "track not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, track)
}

func (s *TrackStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "track not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}

// Device is a device.
type Device struct {
	ID int64 `json:"id"`
	Name string `json:"name"`
	SerialNumber string `json:"serial_number"`
	Firmware string `json:"firmware"`
	Online bool `json:"online"`
	LastSeen time.Time `json:"last_seen"`
}

// DeviceStore keeps devices in memory.
type DeviceStore struct {
	mu     sync.RWMutex
	nextID int64
	items  map[int64]Device
}

// NewDeviceStore returns an empty store.
func NewDeviceStore() *DeviceStore {
	return &DeviceStore{nextID: 1, items: make(map[int64]Device)}
}

// Register adds the devices routes to mux.
func (s *DeviceStore) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /devices", s.list)
	mux.HandleFunc("POST /devices", s.create)
	mux.HandleFunc("GET /devices/{id}", s.get)
	mux.HandleFunc("DELETE /devices/{id}", s.delete)
}

func (s *DeviceStore) list(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := make([]Device, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *DeviceStore) create(w http.ResponseWriter, r *http.Request) {
	var device Device
	if err := json.NewDecoder(r.Body).Decode(&device); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	device.ID = s.nextID
	s.nextID++
	s.items[device.ID] = device
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, device)
}

func (s *DeviceStore) get(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.RLock()
	device, ok := s.items[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "device not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, device)
}

func (s *DeviceStore) delete(w http.ResponseWriter, r *http.Request) {
	id, err := parseID(r)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		http.Error(w, "device not found", http.StatusNotFound)
		return
	}
	delete(s.items, id)
	w.WriteHeader(http.StatusNoContent)
}
