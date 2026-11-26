// Typed client for the REST API. Every resource has list, get, create, update and remove.

export interface Page<T> {
  items: T[];
  page: number;
  pageSize: number;
  total: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

const BASE_URL = '/api/v1';

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export interface User {
  id: number;
  username: string;
  email: string;
  fullName: string;
  isActive: boolean;
  createdAt: string;
}

export type NewUser = Omit<User, 'id'>;

export const usersApi = {
  list(page = 1, pageSize = 20): Promise<Page<User>> {
    return request<Page<User>>('GET', `/users?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<User> {
    return request<User>('GET', `/users/${id}`);
  },
  create(data: NewUser): Promise<User> {
    return request<User>('POST', '/users', data);
  },
  update(id: number, changes: Partial<NewUser>): Promise<User> {
    return request<User>('PATCH', `/users/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/users/${id}`);
  },
};

export interface Product {
  id: number;
  sku: string;
  name: string;
  description: string;
  price: number;
  stock: number;
}

export type NewProduct = Omit<Product, 'id'>;

export const productsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Product>> {
    return request<Page<Product>>('GET', `/products?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Product> {
    return request<Product>('GET', `/products/${id}`);
  },
  create(data: NewProduct): Promise<Product> {
    return request<Product>('POST', '/products', data);
  },
  update(id: number, changes: Partial<NewProduct>): Promise<Product> {
    return request<Product>('PATCH', `/products/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/products/${id}`);
  },
};

export interface Order {
  id: number;
  customerId: number;
  status: string;
  total: number;
  currency: string;
  placedAt: string;
}

export type NewOrder = Omit<Order, 'id'>;

export const ordersApi = {
  list(page = 1, pageSize = 20): Promise<Page<Order>> {
    return request<Page<Order>>('GET', `/orders?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Order> {
    return request<Order>('GET', `/orders/${id}`);
  },
  create(data: NewOrder): Promise<Order> {
    return request<Order>('POST', '/orders', data);
  },
  update(id: number, changes: Partial<NewOrder>): Promise<Order> {
    return request<Order>('PATCH', `/orders/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/orders/${id}`);
  },
};

export interface OrderItem {
  id: number;
  orderId: number;
  productId: number;
  quantity: number;
  unitPrice: number;
}

export type NewOrderItem = Omit<OrderItem, 'id'>;

export const orderItemsApi = {
  list(page = 1, pageSize = 20): Promise<Page<OrderItem>> {
    return request<Page<OrderItem>>('GET', `/order-items?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<OrderItem> {
    return request<OrderItem>('GET', `/order-items/${id}`);
  },
  create(data: NewOrderItem): Promise<OrderItem> {
    return request<OrderItem>('POST', '/order-items', data);
  },
  update(id: number, changes: Partial<NewOrderItem>): Promise<OrderItem> {
    return request<OrderItem>('PATCH', `/order-items/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/order-items/${id}`);
  },
};

export interface Customer {
  id: number;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  country: string;
}

export type NewCustomer = Omit<Customer, 'id'>;

export const customersApi = {
  list(page = 1, pageSize = 20): Promise<Page<Customer>> {
    return request<Page<Customer>>('GET', `/customers?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Customer> {
    return request<Customer>('GET', `/customers/${id}`);
  },
  create(data: NewCustomer): Promise<Customer> {
    return request<Customer>('POST', '/customers', data);
  },
  update(id: number, changes: Partial<NewCustomer>): Promise<Customer> {
    return request<Customer>('PATCH', `/customers/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/customers/${id}`);
  },
};

export interface Invoice {
  id: number;
  orderId: number;
  number: string;
  amount: number;
  dueDate: string;
  paid: boolean;
}

export type NewInvoice = Omit<Invoice, 'id'>;

export const invoicesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Invoice>> {
    return request<Page<Invoice>>('GET', `/invoices?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Invoice> {
    return request<Invoice>('GET', `/invoices/${id}`);
  },
  create(data: NewInvoice): Promise<Invoice> {
    return request<Invoice>('POST', '/invoices', data);
  },
  update(id: number, changes: Partial<NewInvoice>): Promise<Invoice> {
    return request<Invoice>('PATCH', `/invoices/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/invoices/${id}`);
  },
};

export interface Payment {
  id: number;
  invoiceId: number;
  method: string;
  amount: number;
  reference: string;
  receivedAt: string;
}

export type NewPayment = Omit<Payment, 'id'>;

export const paymentsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Payment>> {
    return request<Page<Payment>>('GET', `/payments?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Payment> {
    return request<Payment>('GET', `/payments/${id}`);
  },
  create(data: NewPayment): Promise<Payment> {
    return request<Payment>('POST', '/payments', data);
  },
  update(id: number, changes: Partial<NewPayment>): Promise<Payment> {
    return request<Payment>('PATCH', `/payments/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/payments/${id}`);
  },
};

export interface Category {
  id: number;
  name: string;
  slug: string;
  parentId: number;
  position: number;
}

export type NewCategory = Omit<Category, 'id'>;

export const categoriesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Category>> {
    return request<Page<Category>>('GET', `/categories?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Category> {
    return request<Category>('GET', `/categories/${id}`);
  },
  create(data: NewCategory): Promise<Category> {
    return request<Category>('POST', '/categories', data);
  },
  update(id: number, changes: Partial<NewCategory>): Promise<Category> {
    return request<Category>('PATCH', `/categories/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/categories/${id}`);
  },
};

export interface Review {
  id: number;
  productId: number;
  userId: number;
  rating: number;
  title: string;
  body: string;
}

export type NewReview = Omit<Review, 'id'>;

export const reviewsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Review>> {
    return request<Page<Review>>('GET', `/reviews?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Review> {
    return request<Review>('GET', `/reviews/${id}`);
  },
  create(data: NewReview): Promise<Review> {
    return request<Review>('POST', '/reviews', data);
  },
  update(id: number, changes: Partial<NewReview>): Promise<Review> {
    return request<Review>('PATCH', `/reviews/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/reviews/${id}`);
  },
};

export interface Address {
  id: number;
  customerId: number;
  street: string;
  city: string;
  postalCode: string;
  country: string;
}

export type NewAddress = Omit<Address, 'id'>;

export const addressesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Address>> {
    return request<Page<Address>>('GET', `/addresses?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Address> {
    return request<Address>('GET', `/addresses/${id}`);
  },
  create(data: NewAddress): Promise<Address> {
    return request<Address>('POST', '/addresses', data);
  },
  update(id: number, changes: Partial<NewAddress>): Promise<Address> {
    return request<Address>('PATCH', `/addresses/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/addresses/${id}`);
  },
};

export interface Shipment {
  id: number;
  orderId: number;
  carrier: string;
  trackingNumber: string;
  shippedAt: string;
  delivered: boolean;
}

export type NewShipment = Omit<Shipment, 'id'>;

export const shipmentsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Shipment>> {
    return request<Page<Shipment>>('GET', `/shipments?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Shipment> {
    return request<Shipment>('GET', `/shipments/${id}`);
  },
  create(data: NewShipment): Promise<Shipment> {
    return request<Shipment>('POST', '/shipments', data);
  },
  update(id: number, changes: Partial<NewShipment>): Promise<Shipment> {
    return request<Shipment>('PATCH', `/shipments/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/shipments/${id}`);
  },
};

export interface Warehouse {
  id: number;
  name: string;
  city: string;
  capacity: number;
  isOpen: boolean;
}

export type NewWarehouse = Omit<Warehouse, 'id'>;

export const warehousesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Warehouse>> {
    return request<Page<Warehouse>>('GET', `/warehouses?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Warehouse> {
    return request<Warehouse>('GET', `/warehouses/${id}`);
  },
  create(data: NewWarehouse): Promise<Warehouse> {
    return request<Warehouse>('POST', '/warehouses', data);
  },
  update(id: number, changes: Partial<NewWarehouse>): Promise<Warehouse> {
    return request<Warehouse>('PATCH', `/warehouses/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/warehouses/${id}`);
  },
};

export interface Supplier {
  id: number;
  name: string;
  contactEmail: string;
  phone: string;
  country: string;
  rating: number;
}

export type NewSupplier = Omit<Supplier, 'id'>;

export const suppliersApi = {
  list(page = 1, pageSize = 20): Promise<Page<Supplier>> {
    return request<Page<Supplier>>('GET', `/suppliers?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Supplier> {
    return request<Supplier>('GET', `/suppliers/${id}`);
  },
  create(data: NewSupplier): Promise<Supplier> {
    return request<Supplier>('POST', '/suppliers', data);
  },
  update(id: number, changes: Partial<NewSupplier>): Promise<Supplier> {
    return request<Supplier>('PATCH', `/suppliers/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/suppliers/${id}`);
  },
};

export interface Coupon {
  id: number;
  code: string;
  discountPercent: number;
  validUntil: string;
  maxUses: number;
  isActive: boolean;
}

export type NewCoupon = Omit<Coupon, 'id'>;

export const couponsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Coupon>> {
    return request<Page<Coupon>>('GET', `/coupons?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Coupon> {
    return request<Coupon>('GET', `/coupons/${id}`);
  },
  create(data: NewCoupon): Promise<Coupon> {
    return request<Coupon>('POST', '/coupons', data);
  },
  update(id: number, changes: Partial<NewCoupon>): Promise<Coupon> {
    return request<Coupon>('PATCH', `/coupons/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/coupons/${id}`);
  },
};

export interface Employee {
  id: number;
  firstName: string;
  lastName: string;
  email: string;
  department: string;
  salary: number;
}

export type NewEmployee = Omit<Employee, 'id'>;

export const employeesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Employee>> {
    return request<Page<Employee>>('GET', `/employees?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Employee> {
    return request<Employee>('GET', `/employees/${id}`);
  },
  create(data: NewEmployee): Promise<Employee> {
    return request<Employee>('POST', '/employees', data);
  },
  update(id: number, changes: Partial<NewEmployee>): Promise<Employee> {
    return request<Employee>('PATCH', `/employees/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/employees/${id}`);
  },
};

export interface Department {
  id: number;
  name: string;
  code: string;
  managerId: number;
  budget: number;
}

export type NewDepartment = Omit<Department, 'id'>;

export const departmentsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Department>> {
    return request<Page<Department>>('GET', `/departments?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Department> {
    return request<Department>('GET', `/departments/${id}`);
  },
  create(data: NewDepartment): Promise<Department> {
    return request<Department>('POST', '/departments', data);
  },
  update(id: number, changes: Partial<NewDepartment>): Promise<Department> {
    return request<Department>('PATCH', `/departments/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/departments/${id}`);
  },
};

export interface Project {
  id: number;
  name: string;
  description: string;
  startDate: string;
  endDate: string;
  budget: number;
}

export type NewProject = Omit<Project, 'id'>;

export const projectsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Project>> {
    return request<Page<Project>>('GET', `/projects?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Project> {
    return request<Project>('GET', `/projects/${id}`);
  },
  create(data: NewProject): Promise<Project> {
    return request<Project>('POST', '/projects', data);
  },
  update(id: number, changes: Partial<NewProject>): Promise<Project> {
    return request<Project>('PATCH', `/projects/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/projects/${id}`);
  },
};

export interface Task {
  id: number;
  projectId: number;
  title: string;
  status: string;
  priority: number;
  dueDate: string;
}

export type NewTask = Omit<Task, 'id'>;

export const tasksApi = {
  list(page = 1, pageSize = 20): Promise<Page<Task>> {
    return request<Page<Task>>('GET', `/tasks?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Task> {
    return request<Task>('GET', `/tasks/${id}`);
  },
  create(data: NewTask): Promise<Task> {
    return request<Task>('POST', '/tasks', data);
  },
  update(id: number, changes: Partial<NewTask>): Promise<Task> {
    return request<Task>('PATCH', `/tasks/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/tasks/${id}`);
  },
};

export interface Comment {
  id: number;
  taskId: number;
  authorId: number;
  body: string;
  createdAt: string;
}

export type NewComment = Omit<Comment, 'id'>;

export const commentsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Comment>> {
    return request<Page<Comment>>('GET', `/comments?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Comment> {
    return request<Comment>('GET', `/comments/${id}`);
  },
  create(data: NewComment): Promise<Comment> {
    return request<Comment>('POST', '/comments', data);
  },
  update(id: number, changes: Partial<NewComment>): Promise<Comment> {
    return request<Comment>('PATCH', `/comments/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/comments/${id}`);
  },
};

export interface Tag {
  id: number;
  name: string;
  color: string;
  usageCount: number;
}

export type NewTag = Omit<Tag, 'id'>;

export const tagsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Tag>> {
    return request<Page<Tag>>('GET', `/tags?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Tag> {
    return request<Tag>('GET', `/tags/${id}`);
  },
  create(data: NewTag): Promise<Tag> {
    return request<Tag>('POST', '/tags', data);
  },
  update(id: number, changes: Partial<NewTag>): Promise<Tag> {
    return request<Tag>('PATCH', `/tags/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/tags/${id}`);
  },
};

export interface Article {
  id: number;
  title: string;
  slug: string;
  body: string;
  authorId: number;
  published: boolean;
}

export type NewArticle = Omit<Article, 'id'>;

export const articlesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Article>> {
    return request<Page<Article>>('GET', `/articles?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Article> {
    return request<Article>('GET', `/articles/${id}`);
  },
  create(data: NewArticle): Promise<Article> {
    return request<Article>('POST', '/articles', data);
  },
  update(id: number, changes: Partial<NewArticle>): Promise<Article> {
    return request<Article>('PATCH', `/articles/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/articles/${id}`);
  },
};

export interface Author {
  id: number;
  name: string;
  bio: string;
  email: string;
  website: string;
}

export type NewAuthor = Omit<Author, 'id'>;

export const authorsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Author>> {
    return request<Page<Author>>('GET', `/authors?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Author> {
    return request<Author>('GET', `/authors/${id}`);
  },
  create(data: NewAuthor): Promise<Author> {
    return request<Author>('POST', '/authors', data);
  },
  update(id: number, changes: Partial<NewAuthor>): Promise<Author> {
    return request<Author>('PATCH', `/authors/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/authors/${id}`);
  },
};

export interface Book {
  id: number;
  title: string;
  isbn: string;
  authorId: number;
  year: number;
  pages: number;
}

export type NewBook = Omit<Book, 'id'>;

export const booksApi = {
  list(page = 1, pageSize = 20): Promise<Page<Book>> {
    return request<Page<Book>>('GET', `/books?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Book> {
    return request<Book>('GET', `/books/${id}`);
  },
  create(data: NewBook): Promise<Book> {
    return request<Book>('POST', '/books', data);
  },
  update(id: number, changes: Partial<NewBook>): Promise<Book> {
    return request<Book>('PATCH', `/books/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/books/${id}`);
  },
};

export interface Library {
  id: number;
  name: string;
  city: string;
  openingHour: number;
  closingHour: number;
}

export type NewLibrary = Omit<Library, 'id'>;

export const librariesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Library>> {
    return request<Page<Library>>('GET', `/libraries?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Library> {
    return request<Library>('GET', `/libraries/${id}`);
  },
  create(data: NewLibrary): Promise<Library> {
    return request<Library>('POST', '/libraries', data);
  },
  update(id: number, changes: Partial<NewLibrary>): Promise<Library> {
    return request<Library>('PATCH', `/libraries/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/libraries/${id}`);
  },
};

export interface Loan {
  id: number;
  bookId: number;
  memberId: number;
  borrowedAt: string;
  dueDate: string;
  returned: boolean;
}

export type NewLoan = Omit<Loan, 'id'>;

export const loansApi = {
  list(page = 1, pageSize = 20): Promise<Page<Loan>> {
    return request<Page<Loan>>('GET', `/loans?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Loan> {
    return request<Loan>('GET', `/loans/${id}`);
  },
  create(data: NewLoan): Promise<Loan> {
    return request<Loan>('POST', '/loans', data);
  },
  update(id: number, changes: Partial<NewLoan>): Promise<Loan> {
    return request<Loan>('PATCH', `/loans/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/loans/${id}`);
  },
};

export interface Member {
  id: number;
  firstName: string;
  lastName: string;
  email: string;
  joinedAt: string;
}

export type NewMember = Omit<Member, 'id'>;

export const membersApi = {
  list(page = 1, pageSize = 20): Promise<Page<Member>> {
    return request<Page<Member>>('GET', `/members?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Member> {
    return request<Member>('GET', `/members/${id}`);
  },
  create(data: NewMember): Promise<Member> {
    return request<Member>('POST', '/members', data);
  },
  update(id: number, changes: Partial<NewMember>): Promise<Member> {
    return request<Member>('PATCH', `/members/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/members/${id}`);
  },
};

export interface Event {
  id: number;
  title: string;
  location: string;
  startsAt: string;
  endsAt: string;
  capacity: number;
}

export type NewEvent = Omit<Event, 'id'>;

export const eventsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Event>> {
    return request<Page<Event>>('GET', `/events?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Event> {
    return request<Event>('GET', `/events/${id}`);
  },
  create(data: NewEvent): Promise<Event> {
    return request<Event>('POST', '/events', data);
  },
  update(id: number, changes: Partial<NewEvent>): Promise<Event> {
    return request<Event>('PATCH', `/events/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/events/${id}`);
  },
};

export interface Ticket {
  id: number;
  eventId: number;
  holderName: string;
  seat: string;
  price: number;
  checkedIn: boolean;
}

export type NewTicket = Omit<Ticket, 'id'>;

export const ticketsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Ticket>> {
    return request<Page<Ticket>>('GET', `/tickets?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Ticket> {
    return request<Ticket>('GET', `/tickets/${id}`);
  },
  create(data: NewTicket): Promise<Ticket> {
    return request<Ticket>('POST', '/tickets', data);
  },
  update(id: number, changes: Partial<NewTicket>): Promise<Ticket> {
    return request<Ticket>('PATCH', `/tickets/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/tickets/${id}`);
  },
};

export interface Venue {
  id: number;
  name: string;
  address: string;
  city: string;
  capacity: number;
}

export type NewVenue = Omit<Venue, 'id'>;

export const venuesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Venue>> {
    return request<Page<Venue>>('GET', `/venues?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Venue> {
    return request<Venue>('GET', `/venues/${id}`);
  },
  create(data: NewVenue): Promise<Venue> {
    return request<Venue>('POST', '/venues', data);
  },
  update(id: number, changes: Partial<NewVenue>): Promise<Venue> {
    return request<Venue>('PATCH', `/venues/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/venues/${id}`);
  },
};

export interface Vehicle {
  id: number;
  plate: string;
  make: string;
  model: string;
  year: number;
  mileage: number;
}

export type NewVehicle = Omit<Vehicle, 'id'>;

export const vehiclesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Vehicle>> {
    return request<Page<Vehicle>>('GET', `/vehicles?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Vehicle> {
    return request<Vehicle>('GET', `/vehicles/${id}`);
  },
  create(data: NewVehicle): Promise<Vehicle> {
    return request<Vehicle>('POST', '/vehicles', data);
  },
  update(id: number, changes: Partial<NewVehicle>): Promise<Vehicle> {
    return request<Vehicle>('PATCH', `/vehicles/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/vehicles/${id}`);
  },
};

export interface Driver {
  id: number;
  firstName: string;
  lastName: string;
  licenseNumber: string;
  rating: number;
}

export type NewDriver = Omit<Driver, 'id'>;

export const driversApi = {
  list(page = 1, pageSize = 20): Promise<Page<Driver>> {
    return request<Page<Driver>>('GET', `/drivers?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Driver> {
    return request<Driver>('GET', `/drivers/${id}`);
  },
  create(data: NewDriver): Promise<Driver> {
    return request<Driver>('POST', '/drivers', data);
  },
  update(id: number, changes: Partial<NewDriver>): Promise<Driver> {
    return request<Driver>('PATCH', `/drivers/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/drivers/${id}`);
  },
};

export interface Trip {
  id: number;
  vehicleId: number;
  driverId: number;
  distanceKm: number;
  startedAt: string;
  fare: number;
}

export type NewTrip = Omit<Trip, 'id'>;

export const tripsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Trip>> {
    return request<Page<Trip>>('GET', `/trips?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Trip> {
    return request<Trip>('GET', `/trips/${id}`);
  },
  create(data: NewTrip): Promise<Trip> {
    return request<Trip>('POST', '/trips', data);
  },
  update(id: number, changes: Partial<NewTrip>): Promise<Trip> {
    return request<Trip>('PATCH', `/trips/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/trips/${id}`);
  },
};

export interface Patient {
  id: number;
  firstName: string;
  lastName: string;
  birthDate: string;
  phone: string;
}

export type NewPatient = Omit<Patient, 'id'>;

export const patientsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Patient>> {
    return request<Page<Patient>>('GET', `/patients?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Patient> {
    return request<Patient>('GET', `/patients/${id}`);
  },
  create(data: NewPatient): Promise<Patient> {
    return request<Patient>('POST', '/patients', data);
  },
  update(id: number, changes: Partial<NewPatient>): Promise<Patient> {
    return request<Patient>('PATCH', `/patients/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/patients/${id}`);
  },
};

export interface Doctor {
  id: number;
  firstName: string;
  lastName: string;
  specialty: string;
  room: string;
}

export type NewDoctor = Omit<Doctor, 'id'>;

export const doctorsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Doctor>> {
    return request<Page<Doctor>>('GET', `/doctors?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Doctor> {
    return request<Doctor>('GET', `/doctors/${id}`);
  },
  create(data: NewDoctor): Promise<Doctor> {
    return request<Doctor>('POST', '/doctors', data);
  },
  update(id: number, changes: Partial<NewDoctor>): Promise<Doctor> {
    return request<Doctor>('PATCH', `/doctors/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/doctors/${id}`);
  },
};

export interface Appointment {
  id: number;
  patientId: number;
  doctorId: number;
  scheduledAt: string;
  reason: string;
  confirmed: boolean;
}

export type NewAppointment = Omit<Appointment, 'id'>;

export const appointmentsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Appointment>> {
    return request<Page<Appointment>>('GET', `/appointments?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Appointment> {
    return request<Appointment>('GET', `/appointments/${id}`);
  },
  create(data: NewAppointment): Promise<Appointment> {
    return request<Appointment>('POST', '/appointments', data);
  },
  update(id: number, changes: Partial<NewAppointment>): Promise<Appointment> {
    return request<Appointment>('PATCH', `/appointments/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/appointments/${id}`);
  },
};

export interface Course {
  id: number;
  title: string;
  code: string;
  credits: number;
  teacherId: number;
}

export type NewCourse = Omit<Course, 'id'>;

export const coursesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Course>> {
    return request<Page<Course>>('GET', `/courses?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Course> {
    return request<Course>('GET', `/courses/${id}`);
  },
  create(data: NewCourse): Promise<Course> {
    return request<Course>('POST', '/courses', data);
  },
  update(id: number, changes: Partial<NewCourse>): Promise<Course> {
    return request<Course>('PATCH', `/courses/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/courses/${id}`);
  },
};

export interface Student {
  id: number;
  firstName: string;
  lastName: string;
  email: string;
  enrolledAt: string;
}

export type NewStudent = Omit<Student, 'id'>;

export const studentsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Student>> {
    return request<Page<Student>>('GET', `/students?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Student> {
    return request<Student>('GET', `/students/${id}`);
  },
  create(data: NewStudent): Promise<Student> {
    return request<Student>('POST', '/students', data);
  },
  update(id: number, changes: Partial<NewStudent>): Promise<Student> {
    return request<Student>('PATCH', `/students/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/students/${id}`);
  },
};

export interface Enrollment {
  id: number;
  studentId: number;
  courseId: number;
  grade: number;
  completed: boolean;
}

export type NewEnrollment = Omit<Enrollment, 'id'>;

export const enrollmentsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Enrollment>> {
    return request<Page<Enrollment>>('GET', `/enrollments?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Enrollment> {
    return request<Enrollment>('GET', `/enrollments/${id}`);
  },
  create(data: NewEnrollment): Promise<Enrollment> {
    return request<Enrollment>('POST', '/enrollments', data);
  },
  update(id: number, changes: Partial<NewEnrollment>): Promise<Enrollment> {
    return request<Enrollment>('PATCH', `/enrollments/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/enrollments/${id}`);
  },
};

export interface Room {
  id: number;
  number: string;
  floor: number;
  beds: number;
  pricePerNight: number;
}

export type NewRoom = Omit<Room, 'id'>;

export const roomsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Room>> {
    return request<Page<Room>>('GET', `/rooms?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Room> {
    return request<Room>('GET', `/rooms/${id}`);
  },
  create(data: NewRoom): Promise<Room> {
    return request<Room>('POST', '/rooms', data);
  },
  update(id: number, changes: Partial<NewRoom>): Promise<Room> {
    return request<Room>('PATCH', `/rooms/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/rooms/${id}`);
  },
};

export interface Booking {
  id: number;
  roomId: number;
  guestName: string;
  checkIn: string;
  checkOut: string;
  paid: boolean;
}

export type NewBooking = Omit<Booking, 'id'>;

export const bookingsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Booking>> {
    return request<Page<Booking>>('GET', `/bookings?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Booking> {
    return request<Booking>('GET', `/bookings/${id}`);
  },
  create(data: NewBooking): Promise<Booking> {
    return request<Booking>('POST', '/bookings', data);
  },
  update(id: number, changes: Partial<NewBooking>): Promise<Booking> {
    return request<Booking>('PATCH', `/bookings/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/bookings/${id}`);
  },
};

export interface Subscription {
  id: number;
  userId: number;
  plan: string;
  price: number;
  renewsAt: string;
  isActive: boolean;
}

export type NewSubscription = Omit<Subscription, 'id'>;

export const subscriptionsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Subscription>> {
    return request<Page<Subscription>>('GET', `/subscriptions?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Subscription> {
    return request<Subscription>('GET', `/subscriptions/${id}`);
  },
  create(data: NewSubscription): Promise<Subscription> {
    return request<Subscription>('POST', '/subscriptions', data);
  },
  update(id: number, changes: Partial<NewSubscription>): Promise<Subscription> {
    return request<Subscription>('PATCH', `/subscriptions/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/subscriptions/${id}`);
  },
};

export interface Notification {
  id: number;
  userId: number;
  title: string;
  message: string;
  read: boolean;
  sentAt: string;
}

export type NewNotification = Omit<Notification, 'id'>;

export const notificationsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Notification>> {
    return request<Page<Notification>>('GET', `/notifications?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Notification> {
    return request<Notification>('GET', `/notifications/${id}`);
  },
  create(data: NewNotification): Promise<Notification> {
    return request<Notification>('POST', '/notifications', data);
  },
  update(id: number, changes: Partial<NewNotification>): Promise<Notification> {
    return request<Notification>('PATCH', `/notifications/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/notifications/${id}`);
  },
};

export interface Session {
  id: number;
  userId: number;
  token: string;
  expiresAt: string;
  ipAddress: string;
}

export type NewSession = Omit<Session, 'id'>;

export const sessionsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Session>> {
    return request<Page<Session>>('GET', `/sessions?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Session> {
    return request<Session>('GET', `/sessions/${id}`);
  },
  create(data: NewSession): Promise<Session> {
    return request<Session>('POST', '/sessions', data);
  },
  update(id: number, changes: Partial<NewSession>): Promise<Session> {
    return request<Session>('PATCH', `/sessions/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/sessions/${id}`);
  },
};

export interface AuditLog {
  id: number;
  actorId: number;
  action: string;
  target: string;
  createdAt: string;
}

export type NewAuditLog = Omit<AuditLog, 'id'>;

export const auditLogsApi = {
  list(page = 1, pageSize = 20): Promise<Page<AuditLog>> {
    return request<Page<AuditLog>>('GET', `/audit-logs?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<AuditLog> {
    return request<AuditLog>('GET', `/audit-logs/${id}`);
  },
  create(data: NewAuditLog): Promise<AuditLog> {
    return request<AuditLog>('POST', '/audit-logs', data);
  },
  update(id: number, changes: Partial<NewAuditLog>): Promise<AuditLog> {
    return request<AuditLog>('PATCH', `/audit-logs/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/audit-logs/${id}`);
  },
};

export interface Setting {
  id: number;
  key: string;
  value: string;
  description: string;
}

export type NewSetting = Omit<Setting, 'id'>;

export const settingsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Setting>> {
    return request<Page<Setting>>('GET', `/settings?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Setting> {
    return request<Setting>('GET', `/settings/${id}`);
  },
  create(data: NewSetting): Promise<Setting> {
    return request<Setting>('POST', '/settings', data);
  },
  update(id: number, changes: Partial<NewSetting>): Promise<Setting> {
    return request<Setting>('PATCH', `/settings/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/settings/${id}`);
  },
};

export interface Recipe {
  id: number;
  title: string;
  servings: number;
  prepMinutes: number;
  cookMinutes: number;
  vegetarian: boolean;
}

export type NewRecipe = Omit<Recipe, 'id'>;

export const recipesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Recipe>> {
    return request<Page<Recipe>>('GET', `/recipes?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Recipe> {
    return request<Recipe>('GET', `/recipes/${id}`);
  },
  create(data: NewRecipe): Promise<Recipe> {
    return request<Recipe>('POST', '/recipes', data);
  },
  update(id: number, changes: Partial<NewRecipe>): Promise<Recipe> {
    return request<Recipe>('PATCH', `/recipes/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/recipes/${id}`);
  },
};

export interface Ingredient {
  id: number;
  recipeId: number;
  name: string;
  amount: number;
  unit: string;
}

export type NewIngredient = Omit<Ingredient, 'id'>;

export const ingredientsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Ingredient>> {
    return request<Page<Ingredient>>('GET', `/ingredients?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Ingredient> {
    return request<Ingredient>('GET', `/ingredients/${id}`);
  },
  create(data: NewIngredient): Promise<Ingredient> {
    return request<Ingredient>('POST', '/ingredients', data);
  },
  update(id: number, changes: Partial<NewIngredient>): Promise<Ingredient> {
    return request<Ingredient>('PATCH', `/ingredients/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/ingredients/${id}`);
  },
};

export interface Playlist {
  id: number;
  name: string;
  ownerId: number;
  isPublic: boolean;
  trackCount: number;
}

export type NewPlaylist = Omit<Playlist, 'id'>;

export const playlistsApi = {
  list(page = 1, pageSize = 20): Promise<Page<Playlist>> {
    return request<Page<Playlist>>('GET', `/playlists?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Playlist> {
    return request<Playlist>('GET', `/playlists/${id}`);
  },
  create(data: NewPlaylist): Promise<Playlist> {
    return request<Playlist>('POST', '/playlists', data);
  },
  update(id: number, changes: Partial<NewPlaylist>): Promise<Playlist> {
    return request<Playlist>('PATCH', `/playlists/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/playlists/${id}`);
  },
};

export interface Track {
  id: number;
  title: string;
  artist: string;
  album: string;
  durationSeconds: number;
}

export type NewTrack = Omit<Track, 'id'>;

export const tracksApi = {
  list(page = 1, pageSize = 20): Promise<Page<Track>> {
    return request<Page<Track>>('GET', `/tracks?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Track> {
    return request<Track>('GET', `/tracks/${id}`);
  },
  create(data: NewTrack): Promise<Track> {
    return request<Track>('POST', '/tracks', data);
  },
  update(id: number, changes: Partial<NewTrack>): Promise<Track> {
    return request<Track>('PATCH', `/tracks/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/tracks/${id}`);
  },
};

export interface Device {
  id: number;
  name: string;
  serialNumber: string;
  firmware: string;
  online: boolean;
  lastSeen: string;
}

export type NewDevice = Omit<Device, 'id'>;

export const devicesApi = {
  list(page = 1, pageSize = 20): Promise<Page<Device>> {
    return request<Page<Device>>('GET', `/devices?page=${page}&pageSize=${pageSize}`);
  },
  get(id: number): Promise<Device> {
    return request<Device>('GET', `/devices/${id}`);
  },
  create(data: NewDevice): Promise<Device> {
    return request<Device>('POST', '/devices', data);
  },
  update(id: number, changes: Partial<NewDevice>): Promise<Device> {
    return request<Device>('PATCH', `/devices/${id}`, changes);
  },
  remove(id: number): Promise<void> {
    return request<void>('DELETE', `/devices/${id}`);
  },
};
