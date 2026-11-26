// Express routes with input checks, one router per resource.
const express = require('express');

function asyncHandler(fn) {
  return (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
}

function userRouter(db) {
  const router = express.Router();

  router.get('/users', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const users = await db('users').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: users, page, pageSize });
  }));

  router.get('/users/:id', asyncHandler(async (req, res) => {
    const user = await db('users').where({ id: req.params.id }).first();
    if (!user) {
      return res.status(404).json({ error: 'user not found' });
    }
    res.json(user);
  }));

  router.post('/users', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.username !== 'string' || req.body.username.length === 0) {
      errors.push('username is required');
    }
    if (typeof req.body.email !== 'string' || req.body.email.length === 0) {
      errors.push('email is required');
    }
    if (typeof req.body.full_name !== 'string' || req.body.full_name.length === 0) {
      errors.push('full_name is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('users').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/users/:id', asyncHandler(async (req, res) => {
    const removed = await db('users').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'user not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function productRouter(db) {
  const router = express.Router();

  router.get('/products', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const products = await db('products').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: products, page, pageSize });
  }));

  router.get('/products/:id', asyncHandler(async (req, res) => {
    const product = await db('products').where({ id: req.params.id }).first();
    if (!product) {
      return res.status(404).json({ error: 'product not found' });
    }
    res.json(product);
  }));

  router.post('/products', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.sku !== 'string' || req.body.sku.length === 0) {
      errors.push('sku is required');
    }
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.description !== 'string' || req.body.description.length === 0) {
      errors.push('description is required');
    }
    if (typeof req.body.price !== 'number') {
      errors.push('price must be a number');
    }
    if (typeof req.body.stock !== 'number') {
      errors.push('stock must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('products').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/products/:id', asyncHandler(async (req, res) => {
    const removed = await db('products').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'product not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function orderRouter(db) {
  const router = express.Router();

  router.get('/orders', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const orders = await db('orders').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: orders, page, pageSize });
  }));

  router.get('/orders/:id', asyncHandler(async (req, res) => {
    const order = await db('orders').where({ id: req.params.id }).first();
    if (!order) {
      return res.status(404).json({ error: 'order not found' });
    }
    res.json(order);
  }));

  router.post('/orders', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.customer_id !== 'number') {
      errors.push('customer_id must be a number');
    }
    if (typeof req.body.status !== 'string' || req.body.status.length === 0) {
      errors.push('status is required');
    }
    if (typeof req.body.total !== 'number') {
      errors.push('total must be a number');
    }
    if (typeof req.body.currency !== 'string' || req.body.currency.length === 0) {
      errors.push('currency is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('orders').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/orders/:id', asyncHandler(async (req, res) => {
    const removed = await db('orders').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'order not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function orderItemRouter(db) {
  const router = express.Router();

  router.get('/order-items', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const orderItems = await db('order_items').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: orderItems, page, pageSize });
  }));

  router.get('/order-items/:id', asyncHandler(async (req, res) => {
    const orderItem = await db('order_items').where({ id: req.params.id }).first();
    if (!orderItem) {
      return res.status(404).json({ error: 'order item not found' });
    }
    res.json(orderItem);
  }));

  router.post('/order-items', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.order_id !== 'number') {
      errors.push('order_id must be a number');
    }
    if (typeof req.body.product_id !== 'number') {
      errors.push('product_id must be a number');
    }
    if (typeof req.body.quantity !== 'number') {
      errors.push('quantity must be a number');
    }
    if (typeof req.body.unit_price !== 'number') {
      errors.push('unit_price must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('order_items').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/order-items/:id', asyncHandler(async (req, res) => {
    const removed = await db('order_items').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'order item not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function customerRouter(db) {
  const router = express.Router();

  router.get('/customers', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const customers = await db('customers').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: customers, page, pageSize });
  }));

  router.get('/customers/:id', asyncHandler(async (req, res) => {
    const customer = await db('customers').where({ id: req.params.id }).first();
    if (!customer) {
      return res.status(404).json({ error: 'customer not found' });
    }
    res.json(customer);
  }));

  router.post('/customers', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.first_name !== 'string' || req.body.first_name.length === 0) {
      errors.push('first_name is required');
    }
    if (typeof req.body.last_name !== 'string' || req.body.last_name.length === 0) {
      errors.push('last_name is required');
    }
    if (typeof req.body.email !== 'string' || req.body.email.length === 0) {
      errors.push('email is required');
    }
    if (typeof req.body.phone !== 'string' || req.body.phone.length === 0) {
      errors.push('phone is required');
    }
    if (typeof req.body.country !== 'string' || req.body.country.length === 0) {
      errors.push('country is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('customers').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/customers/:id', asyncHandler(async (req, res) => {
    const removed = await db('customers').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'customer not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function invoiceRouter(db) {
  const router = express.Router();

  router.get('/invoices', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const invoices = await db('invoices').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: invoices, page, pageSize });
  }));

  router.get('/invoices/:id', asyncHandler(async (req, res) => {
    const invoice = await db('invoices').where({ id: req.params.id }).first();
    if (!invoice) {
      return res.status(404).json({ error: 'invoice not found' });
    }
    res.json(invoice);
  }));

  router.post('/invoices', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.order_id !== 'number') {
      errors.push('order_id must be a number');
    }
    if (typeof req.body.number !== 'string' || req.body.number.length === 0) {
      errors.push('number is required');
    }
    if (typeof req.body.amount !== 'number') {
      errors.push('amount must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('invoices').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/invoices/:id', asyncHandler(async (req, res) => {
    const removed = await db('invoices').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'invoice not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function paymentRouter(db) {
  const router = express.Router();

  router.get('/payments', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const payments = await db('payments').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: payments, page, pageSize });
  }));

  router.get('/payments/:id', asyncHandler(async (req, res) => {
    const payment = await db('payments').where({ id: req.params.id }).first();
    if (!payment) {
      return res.status(404).json({ error: 'payment not found' });
    }
    res.json(payment);
  }));

  router.post('/payments', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.invoice_id !== 'number') {
      errors.push('invoice_id must be a number');
    }
    if (typeof req.body.method !== 'string' || req.body.method.length === 0) {
      errors.push('method is required');
    }
    if (typeof req.body.amount !== 'number') {
      errors.push('amount must be a number');
    }
    if (typeof req.body.reference !== 'string' || req.body.reference.length === 0) {
      errors.push('reference is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('payments').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/payments/:id', asyncHandler(async (req, res) => {
    const removed = await db('payments').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'payment not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function categoryRouter(db) {
  const router = express.Router();

  router.get('/categories', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const categories = await db('categories').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: categories, page, pageSize });
  }));

  router.get('/categories/:id', asyncHandler(async (req, res) => {
    const category = await db('categories').where({ id: req.params.id }).first();
    if (!category) {
      return res.status(404).json({ error: 'category not found' });
    }
    res.json(category);
  }));

  router.post('/categories', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.slug !== 'string' || req.body.slug.length === 0) {
      errors.push('slug is required');
    }
    if (typeof req.body.parent_id !== 'number') {
      errors.push('parent_id must be a number');
    }
    if (typeof req.body.position !== 'number') {
      errors.push('position must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('categories').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/categories/:id', asyncHandler(async (req, res) => {
    const removed = await db('categories').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'category not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function reviewRouter(db) {
  const router = express.Router();

  router.get('/reviews', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const reviews = await db('reviews').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: reviews, page, pageSize });
  }));

  router.get('/reviews/:id', asyncHandler(async (req, res) => {
    const review = await db('reviews').where({ id: req.params.id }).first();
    if (!review) {
      return res.status(404).json({ error: 'review not found' });
    }
    res.json(review);
  }));

  router.post('/reviews', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.product_id !== 'number') {
      errors.push('product_id must be a number');
    }
    if (typeof req.body.user_id !== 'number') {
      errors.push('user_id must be a number');
    }
    if (typeof req.body.rating !== 'number') {
      errors.push('rating must be a number');
    }
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.body !== 'string' || req.body.body.length === 0) {
      errors.push('body is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('reviews').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/reviews/:id', asyncHandler(async (req, res) => {
    const removed = await db('reviews').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'review not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function addressRouter(db) {
  const router = express.Router();

  router.get('/addresses', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const addresses = await db('addresses').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: addresses, page, pageSize });
  }));

  router.get('/addresses/:id', asyncHandler(async (req, res) => {
    const address = await db('addresses').where({ id: req.params.id }).first();
    if (!address) {
      return res.status(404).json({ error: 'address not found' });
    }
    res.json(address);
  }));

  router.post('/addresses', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.customer_id !== 'number') {
      errors.push('customer_id must be a number');
    }
    if (typeof req.body.street !== 'string' || req.body.street.length === 0) {
      errors.push('street is required');
    }
    if (typeof req.body.city !== 'string' || req.body.city.length === 0) {
      errors.push('city is required');
    }
    if (typeof req.body.postal_code !== 'string' || req.body.postal_code.length === 0) {
      errors.push('postal_code is required');
    }
    if (typeof req.body.country !== 'string' || req.body.country.length === 0) {
      errors.push('country is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('addresses').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/addresses/:id', asyncHandler(async (req, res) => {
    const removed = await db('addresses').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'address not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function shipmentRouter(db) {
  const router = express.Router();

  router.get('/shipments', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const shipments = await db('shipments').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: shipments, page, pageSize });
  }));

  router.get('/shipments/:id', asyncHandler(async (req, res) => {
    const shipment = await db('shipments').where({ id: req.params.id }).first();
    if (!shipment) {
      return res.status(404).json({ error: 'shipment not found' });
    }
    res.json(shipment);
  }));

  router.post('/shipments', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.order_id !== 'number') {
      errors.push('order_id must be a number');
    }
    if (typeof req.body.carrier !== 'string' || req.body.carrier.length === 0) {
      errors.push('carrier is required');
    }
    if (typeof req.body.tracking_number !== 'string' || req.body.tracking_number.length === 0) {
      errors.push('tracking_number is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('shipments').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/shipments/:id', asyncHandler(async (req, res) => {
    const removed = await db('shipments').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'shipment not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function warehouseRouter(db) {
  const router = express.Router();

  router.get('/warehouses', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const warehouses = await db('warehouses').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: warehouses, page, pageSize });
  }));

  router.get('/warehouses/:id', asyncHandler(async (req, res) => {
    const warehouse = await db('warehouses').where({ id: req.params.id }).first();
    if (!warehouse) {
      return res.status(404).json({ error: 'warehouse not found' });
    }
    res.json(warehouse);
  }));

  router.post('/warehouses', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.city !== 'string' || req.body.city.length === 0) {
      errors.push('city is required');
    }
    if (typeof req.body.capacity !== 'number') {
      errors.push('capacity must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('warehouses').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/warehouses/:id', asyncHandler(async (req, res) => {
    const removed = await db('warehouses').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'warehouse not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function supplierRouter(db) {
  const router = express.Router();

  router.get('/suppliers', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const suppliers = await db('suppliers').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: suppliers, page, pageSize });
  }));

  router.get('/suppliers/:id', asyncHandler(async (req, res) => {
    const supplier = await db('suppliers').where({ id: req.params.id }).first();
    if (!supplier) {
      return res.status(404).json({ error: 'supplier not found' });
    }
    res.json(supplier);
  }));

  router.post('/suppliers', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.contact_email !== 'string' || req.body.contact_email.length === 0) {
      errors.push('contact_email is required');
    }
    if (typeof req.body.phone !== 'string' || req.body.phone.length === 0) {
      errors.push('phone is required');
    }
    if (typeof req.body.country !== 'string' || req.body.country.length === 0) {
      errors.push('country is required');
    }
    if (typeof req.body.rating !== 'number') {
      errors.push('rating must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('suppliers').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/suppliers/:id', asyncHandler(async (req, res) => {
    const removed = await db('suppliers').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'supplier not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function couponRouter(db) {
  const router = express.Router();

  router.get('/coupons', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const coupons = await db('coupons').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: coupons, page, pageSize });
  }));

  router.get('/coupons/:id', asyncHandler(async (req, res) => {
    const coupon = await db('coupons').where({ id: req.params.id }).first();
    if (!coupon) {
      return res.status(404).json({ error: 'coupon not found' });
    }
    res.json(coupon);
  }));

  router.post('/coupons', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.code !== 'string' || req.body.code.length === 0) {
      errors.push('code is required');
    }
    if (typeof req.body.discount_percent !== 'number') {
      errors.push('discount_percent must be a number');
    }
    if (typeof req.body.max_uses !== 'number') {
      errors.push('max_uses must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('coupons').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/coupons/:id', asyncHandler(async (req, res) => {
    const removed = await db('coupons').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'coupon not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function employeeRouter(db) {
  const router = express.Router();

  router.get('/employees', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const employees = await db('employees').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: employees, page, pageSize });
  }));

  router.get('/employees/:id', asyncHandler(async (req, res) => {
    const employee = await db('employees').where({ id: req.params.id }).first();
    if (!employee) {
      return res.status(404).json({ error: 'employee not found' });
    }
    res.json(employee);
  }));

  router.post('/employees', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.first_name !== 'string' || req.body.first_name.length === 0) {
      errors.push('first_name is required');
    }
    if (typeof req.body.last_name !== 'string' || req.body.last_name.length === 0) {
      errors.push('last_name is required');
    }
    if (typeof req.body.email !== 'string' || req.body.email.length === 0) {
      errors.push('email is required');
    }
    if (typeof req.body.department !== 'string' || req.body.department.length === 0) {
      errors.push('department is required');
    }
    if (typeof req.body.salary !== 'number') {
      errors.push('salary must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('employees').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/employees/:id', asyncHandler(async (req, res) => {
    const removed = await db('employees').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'employee not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function departmentRouter(db) {
  const router = express.Router();

  router.get('/departments', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const departments = await db('departments').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: departments, page, pageSize });
  }));

  router.get('/departments/:id', asyncHandler(async (req, res) => {
    const department = await db('departments').where({ id: req.params.id }).first();
    if (!department) {
      return res.status(404).json({ error: 'department not found' });
    }
    res.json(department);
  }));

  router.post('/departments', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.code !== 'string' || req.body.code.length === 0) {
      errors.push('code is required');
    }
    if (typeof req.body.manager_id !== 'number') {
      errors.push('manager_id must be a number');
    }
    if (typeof req.body.budget !== 'number') {
      errors.push('budget must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('departments').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/departments/:id', asyncHandler(async (req, res) => {
    const removed = await db('departments').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'department not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function projectRouter(db) {
  const router = express.Router();

  router.get('/projects', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const projects = await db('projects').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: projects, page, pageSize });
  }));

  router.get('/projects/:id', asyncHandler(async (req, res) => {
    const project = await db('projects').where({ id: req.params.id }).first();
    if (!project) {
      return res.status(404).json({ error: 'project not found' });
    }
    res.json(project);
  }));

  router.post('/projects', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.description !== 'string' || req.body.description.length === 0) {
      errors.push('description is required');
    }
    if (typeof req.body.budget !== 'number') {
      errors.push('budget must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('projects').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/projects/:id', asyncHandler(async (req, res) => {
    const removed = await db('projects').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'project not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function taskRouter(db) {
  const router = express.Router();

  router.get('/tasks', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const tasks = await db('tasks').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: tasks, page, pageSize });
  }));

  router.get('/tasks/:id', asyncHandler(async (req, res) => {
    const task = await db('tasks').where({ id: req.params.id }).first();
    if (!task) {
      return res.status(404).json({ error: 'task not found' });
    }
    res.json(task);
  }));

  router.post('/tasks', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.project_id !== 'number') {
      errors.push('project_id must be a number');
    }
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.status !== 'string' || req.body.status.length === 0) {
      errors.push('status is required');
    }
    if (typeof req.body.priority !== 'number') {
      errors.push('priority must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('tasks').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/tasks/:id', asyncHandler(async (req, res) => {
    const removed = await db('tasks').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'task not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function commentRouter(db) {
  const router = express.Router();

  router.get('/comments', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const comments = await db('comments').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: comments, page, pageSize });
  }));

  router.get('/comments/:id', asyncHandler(async (req, res) => {
    const comment = await db('comments').where({ id: req.params.id }).first();
    if (!comment) {
      return res.status(404).json({ error: 'comment not found' });
    }
    res.json(comment);
  }));

  router.post('/comments', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.task_id !== 'number') {
      errors.push('task_id must be a number');
    }
    if (typeof req.body.author_id !== 'number') {
      errors.push('author_id must be a number');
    }
    if (typeof req.body.body !== 'string' || req.body.body.length === 0) {
      errors.push('body is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('comments').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/comments/:id', asyncHandler(async (req, res) => {
    const removed = await db('comments').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'comment not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function tagRouter(db) {
  const router = express.Router();

  router.get('/tags', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const tags = await db('tags').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: tags, page, pageSize });
  }));

  router.get('/tags/:id', asyncHandler(async (req, res) => {
    const tag = await db('tags').where({ id: req.params.id }).first();
    if (!tag) {
      return res.status(404).json({ error: 'tag not found' });
    }
    res.json(tag);
  }));

  router.post('/tags', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.color !== 'string' || req.body.color.length === 0) {
      errors.push('color is required');
    }
    if (typeof req.body.usage_count !== 'number') {
      errors.push('usage_count must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('tags').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/tags/:id', asyncHandler(async (req, res) => {
    const removed = await db('tags').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'tag not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function articleRouter(db) {
  const router = express.Router();

  router.get('/articles', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const articles = await db('articles').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: articles, page, pageSize });
  }));

  router.get('/articles/:id', asyncHandler(async (req, res) => {
    const article = await db('articles').where({ id: req.params.id }).first();
    if (!article) {
      return res.status(404).json({ error: 'article not found' });
    }
    res.json(article);
  }));

  router.post('/articles', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.slug !== 'string' || req.body.slug.length === 0) {
      errors.push('slug is required');
    }
    if (typeof req.body.body !== 'string' || req.body.body.length === 0) {
      errors.push('body is required');
    }
    if (typeof req.body.author_id !== 'number') {
      errors.push('author_id must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('articles').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/articles/:id', asyncHandler(async (req, res) => {
    const removed = await db('articles').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'article not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function authorRouter(db) {
  const router = express.Router();

  router.get('/authors', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const authors = await db('authors').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: authors, page, pageSize });
  }));

  router.get('/authors/:id', asyncHandler(async (req, res) => {
    const author = await db('authors').where({ id: req.params.id }).first();
    if (!author) {
      return res.status(404).json({ error: 'author not found' });
    }
    res.json(author);
  }));

  router.post('/authors', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.bio !== 'string' || req.body.bio.length === 0) {
      errors.push('bio is required');
    }
    if (typeof req.body.email !== 'string' || req.body.email.length === 0) {
      errors.push('email is required');
    }
    if (typeof req.body.website !== 'string' || req.body.website.length === 0) {
      errors.push('website is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('authors').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/authors/:id', asyncHandler(async (req, res) => {
    const removed = await db('authors').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'author not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function bookRouter(db) {
  const router = express.Router();

  router.get('/books', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const books = await db('books').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: books, page, pageSize });
  }));

  router.get('/books/:id', asyncHandler(async (req, res) => {
    const book = await db('books').where({ id: req.params.id }).first();
    if (!book) {
      return res.status(404).json({ error: 'book not found' });
    }
    res.json(book);
  }));

  router.post('/books', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.isbn !== 'string' || req.body.isbn.length === 0) {
      errors.push('isbn is required');
    }
    if (typeof req.body.author_id !== 'number') {
      errors.push('author_id must be a number');
    }
    if (typeof req.body.year !== 'number') {
      errors.push('year must be a number');
    }
    if (typeof req.body.pages !== 'number') {
      errors.push('pages must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('books').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/books/:id', asyncHandler(async (req, res) => {
    const removed = await db('books').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'book not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function libraryRouter(db) {
  const router = express.Router();

  router.get('/libraries', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const libraries = await db('libraries').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: libraries, page, pageSize });
  }));

  router.get('/libraries/:id', asyncHandler(async (req, res) => {
    const library = await db('libraries').where({ id: req.params.id }).first();
    if (!library) {
      return res.status(404).json({ error: 'library not found' });
    }
    res.json(library);
  }));

  router.post('/libraries', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.city !== 'string' || req.body.city.length === 0) {
      errors.push('city is required');
    }
    if (typeof req.body.opening_hour !== 'number') {
      errors.push('opening_hour must be a number');
    }
    if (typeof req.body.closing_hour !== 'number') {
      errors.push('closing_hour must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('libraries').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/libraries/:id', asyncHandler(async (req, res) => {
    const removed = await db('libraries').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'library not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function loanRouter(db) {
  const router = express.Router();

  router.get('/loans', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const loans = await db('loans').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: loans, page, pageSize });
  }));

  router.get('/loans/:id', asyncHandler(async (req, res) => {
    const loan = await db('loans').where({ id: req.params.id }).first();
    if (!loan) {
      return res.status(404).json({ error: 'loan not found' });
    }
    res.json(loan);
  }));

  router.post('/loans', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.book_id !== 'number') {
      errors.push('book_id must be a number');
    }
    if (typeof req.body.member_id !== 'number') {
      errors.push('member_id must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('loans').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/loans/:id', asyncHandler(async (req, res) => {
    const removed = await db('loans').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'loan not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function memberRouter(db) {
  const router = express.Router();

  router.get('/members', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const members = await db('members').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: members, page, pageSize });
  }));

  router.get('/members/:id', asyncHandler(async (req, res) => {
    const member = await db('members').where({ id: req.params.id }).first();
    if (!member) {
      return res.status(404).json({ error: 'member not found' });
    }
    res.json(member);
  }));

  router.post('/members', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.first_name !== 'string' || req.body.first_name.length === 0) {
      errors.push('first_name is required');
    }
    if (typeof req.body.last_name !== 'string' || req.body.last_name.length === 0) {
      errors.push('last_name is required');
    }
    if (typeof req.body.email !== 'string' || req.body.email.length === 0) {
      errors.push('email is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('members').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/members/:id', asyncHandler(async (req, res) => {
    const removed = await db('members').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'member not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function eventRouter(db) {
  const router = express.Router();

  router.get('/events', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const events = await db('events').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: events, page, pageSize });
  }));

  router.get('/events/:id', asyncHandler(async (req, res) => {
    const event = await db('events').where({ id: req.params.id }).first();
    if (!event) {
      return res.status(404).json({ error: 'event not found' });
    }
    res.json(event);
  }));

  router.post('/events', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.location !== 'string' || req.body.location.length === 0) {
      errors.push('location is required');
    }
    if (typeof req.body.capacity !== 'number') {
      errors.push('capacity must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('events').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/events/:id', asyncHandler(async (req, res) => {
    const removed = await db('events').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'event not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function ticketRouter(db) {
  const router = express.Router();

  router.get('/tickets', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const tickets = await db('tickets').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: tickets, page, pageSize });
  }));

  router.get('/tickets/:id', asyncHandler(async (req, res) => {
    const ticket = await db('tickets').where({ id: req.params.id }).first();
    if (!ticket) {
      return res.status(404).json({ error: 'ticket not found' });
    }
    res.json(ticket);
  }));

  router.post('/tickets', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.event_id !== 'number') {
      errors.push('event_id must be a number');
    }
    if (typeof req.body.holder_name !== 'string' || req.body.holder_name.length === 0) {
      errors.push('holder_name is required');
    }
    if (typeof req.body.seat !== 'string' || req.body.seat.length === 0) {
      errors.push('seat is required');
    }
    if (typeof req.body.price !== 'number') {
      errors.push('price must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('tickets').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/tickets/:id', asyncHandler(async (req, res) => {
    const removed = await db('tickets').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'ticket not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function venueRouter(db) {
  const router = express.Router();

  router.get('/venues', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const venues = await db('venues').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: venues, page, pageSize });
  }));

  router.get('/venues/:id', asyncHandler(async (req, res) => {
    const venue = await db('venues').where({ id: req.params.id }).first();
    if (!venue) {
      return res.status(404).json({ error: 'venue not found' });
    }
    res.json(venue);
  }));

  router.post('/venues', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.address !== 'string' || req.body.address.length === 0) {
      errors.push('address is required');
    }
    if (typeof req.body.city !== 'string' || req.body.city.length === 0) {
      errors.push('city is required');
    }
    if (typeof req.body.capacity !== 'number') {
      errors.push('capacity must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('venues').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/venues/:id', asyncHandler(async (req, res) => {
    const removed = await db('venues').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'venue not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function vehicleRouter(db) {
  const router = express.Router();

  router.get('/vehicles', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const vehicles = await db('vehicles').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: vehicles, page, pageSize });
  }));

  router.get('/vehicles/:id', asyncHandler(async (req, res) => {
    const vehicle = await db('vehicles').where({ id: req.params.id }).first();
    if (!vehicle) {
      return res.status(404).json({ error: 'vehicle not found' });
    }
    res.json(vehicle);
  }));

  router.post('/vehicles', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.plate !== 'string' || req.body.plate.length === 0) {
      errors.push('plate is required');
    }
    if (typeof req.body.make !== 'string' || req.body.make.length === 0) {
      errors.push('make is required');
    }
    if (typeof req.body.model !== 'string' || req.body.model.length === 0) {
      errors.push('model is required');
    }
    if (typeof req.body.year !== 'number') {
      errors.push('year must be a number');
    }
    if (typeof req.body.mileage !== 'number') {
      errors.push('mileage must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('vehicles').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/vehicles/:id', asyncHandler(async (req, res) => {
    const removed = await db('vehicles').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'vehicle not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function driverRouter(db) {
  const router = express.Router();

  router.get('/drivers', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const drivers = await db('drivers').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: drivers, page, pageSize });
  }));

  router.get('/drivers/:id', asyncHandler(async (req, res) => {
    const driver = await db('drivers').where({ id: req.params.id }).first();
    if (!driver) {
      return res.status(404).json({ error: 'driver not found' });
    }
    res.json(driver);
  }));

  router.post('/drivers', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.first_name !== 'string' || req.body.first_name.length === 0) {
      errors.push('first_name is required');
    }
    if (typeof req.body.last_name !== 'string' || req.body.last_name.length === 0) {
      errors.push('last_name is required');
    }
    if (typeof req.body.license_number !== 'string' || req.body.license_number.length === 0) {
      errors.push('license_number is required');
    }
    if (typeof req.body.rating !== 'number') {
      errors.push('rating must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('drivers').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/drivers/:id', asyncHandler(async (req, res) => {
    const removed = await db('drivers').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'driver not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function tripRouter(db) {
  const router = express.Router();

  router.get('/trips', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const trips = await db('trips').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: trips, page, pageSize });
  }));

  router.get('/trips/:id', asyncHandler(async (req, res) => {
    const trip = await db('trips').where({ id: req.params.id }).first();
    if (!trip) {
      return res.status(404).json({ error: 'trip not found' });
    }
    res.json(trip);
  }));

  router.post('/trips', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.vehicle_id !== 'number') {
      errors.push('vehicle_id must be a number');
    }
    if (typeof req.body.driver_id !== 'number') {
      errors.push('driver_id must be a number');
    }
    if (typeof req.body.distance_km !== 'number') {
      errors.push('distance_km must be a number');
    }
    if (typeof req.body.fare !== 'number') {
      errors.push('fare must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('trips').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/trips/:id', asyncHandler(async (req, res) => {
    const removed = await db('trips').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'trip not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function patientRouter(db) {
  const router = express.Router();

  router.get('/patients', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const patients = await db('patients').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: patients, page, pageSize });
  }));

  router.get('/patients/:id', asyncHandler(async (req, res) => {
    const patient = await db('patients').where({ id: req.params.id }).first();
    if (!patient) {
      return res.status(404).json({ error: 'patient not found' });
    }
    res.json(patient);
  }));

  router.post('/patients', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.first_name !== 'string' || req.body.first_name.length === 0) {
      errors.push('first_name is required');
    }
    if (typeof req.body.last_name !== 'string' || req.body.last_name.length === 0) {
      errors.push('last_name is required');
    }
    if (typeof req.body.phone !== 'string' || req.body.phone.length === 0) {
      errors.push('phone is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('patients').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/patients/:id', asyncHandler(async (req, res) => {
    const removed = await db('patients').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'patient not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function doctorRouter(db) {
  const router = express.Router();

  router.get('/doctors', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const doctors = await db('doctors').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: doctors, page, pageSize });
  }));

  router.get('/doctors/:id', asyncHandler(async (req, res) => {
    const doctor = await db('doctors').where({ id: req.params.id }).first();
    if (!doctor) {
      return res.status(404).json({ error: 'doctor not found' });
    }
    res.json(doctor);
  }));

  router.post('/doctors', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.first_name !== 'string' || req.body.first_name.length === 0) {
      errors.push('first_name is required');
    }
    if (typeof req.body.last_name !== 'string' || req.body.last_name.length === 0) {
      errors.push('last_name is required');
    }
    if (typeof req.body.specialty !== 'string' || req.body.specialty.length === 0) {
      errors.push('specialty is required');
    }
    if (typeof req.body.room !== 'string' || req.body.room.length === 0) {
      errors.push('room is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('doctors').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/doctors/:id', asyncHandler(async (req, res) => {
    const removed = await db('doctors').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'doctor not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function appointmentRouter(db) {
  const router = express.Router();

  router.get('/appointments', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const appointments = await db('appointments').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: appointments, page, pageSize });
  }));

  router.get('/appointments/:id', asyncHandler(async (req, res) => {
    const appointment = await db('appointments').where({ id: req.params.id }).first();
    if (!appointment) {
      return res.status(404).json({ error: 'appointment not found' });
    }
    res.json(appointment);
  }));

  router.post('/appointments', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.patient_id !== 'number') {
      errors.push('patient_id must be a number');
    }
    if (typeof req.body.doctor_id !== 'number') {
      errors.push('doctor_id must be a number');
    }
    if (typeof req.body.reason !== 'string' || req.body.reason.length === 0) {
      errors.push('reason is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('appointments').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/appointments/:id', asyncHandler(async (req, res) => {
    const removed = await db('appointments').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'appointment not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function courseRouter(db) {
  const router = express.Router();

  router.get('/courses', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const courses = await db('courses').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: courses, page, pageSize });
  }));

  router.get('/courses/:id', asyncHandler(async (req, res) => {
    const course = await db('courses').where({ id: req.params.id }).first();
    if (!course) {
      return res.status(404).json({ error: 'course not found' });
    }
    res.json(course);
  }));

  router.post('/courses', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.code !== 'string' || req.body.code.length === 0) {
      errors.push('code is required');
    }
    if (typeof req.body.credits !== 'number') {
      errors.push('credits must be a number');
    }
    if (typeof req.body.teacher_id !== 'number') {
      errors.push('teacher_id must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('courses').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/courses/:id', asyncHandler(async (req, res) => {
    const removed = await db('courses').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'course not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function studentRouter(db) {
  const router = express.Router();

  router.get('/students', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const students = await db('students').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: students, page, pageSize });
  }));

  router.get('/students/:id', asyncHandler(async (req, res) => {
    const student = await db('students').where({ id: req.params.id }).first();
    if (!student) {
      return res.status(404).json({ error: 'student not found' });
    }
    res.json(student);
  }));

  router.post('/students', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.first_name !== 'string' || req.body.first_name.length === 0) {
      errors.push('first_name is required');
    }
    if (typeof req.body.last_name !== 'string' || req.body.last_name.length === 0) {
      errors.push('last_name is required');
    }
    if (typeof req.body.email !== 'string' || req.body.email.length === 0) {
      errors.push('email is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('students').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/students/:id', asyncHandler(async (req, res) => {
    const removed = await db('students').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'student not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function enrollmentRouter(db) {
  const router = express.Router();

  router.get('/enrollments', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const enrollments = await db('enrollments').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: enrollments, page, pageSize });
  }));

  router.get('/enrollments/:id', asyncHandler(async (req, res) => {
    const enrollment = await db('enrollments').where({ id: req.params.id }).first();
    if (!enrollment) {
      return res.status(404).json({ error: 'enrollment not found' });
    }
    res.json(enrollment);
  }));

  router.post('/enrollments', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.student_id !== 'number') {
      errors.push('student_id must be a number');
    }
    if (typeof req.body.course_id !== 'number') {
      errors.push('course_id must be a number');
    }
    if (typeof req.body.grade !== 'number') {
      errors.push('grade must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('enrollments').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/enrollments/:id', asyncHandler(async (req, res) => {
    const removed = await db('enrollments').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'enrollment not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function roomRouter(db) {
  const router = express.Router();

  router.get('/rooms', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const rooms = await db('rooms').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: rooms, page, pageSize });
  }));

  router.get('/rooms/:id', asyncHandler(async (req, res) => {
    const room = await db('rooms').where({ id: req.params.id }).first();
    if (!room) {
      return res.status(404).json({ error: 'room not found' });
    }
    res.json(room);
  }));

  router.post('/rooms', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.number !== 'string' || req.body.number.length === 0) {
      errors.push('number is required');
    }
    if (typeof req.body.floor !== 'number') {
      errors.push('floor must be a number');
    }
    if (typeof req.body.beds !== 'number') {
      errors.push('beds must be a number');
    }
    if (typeof req.body.price_per_night !== 'number') {
      errors.push('price_per_night must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('rooms').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/rooms/:id', asyncHandler(async (req, res) => {
    const removed = await db('rooms').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'room not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function bookingRouter(db) {
  const router = express.Router();

  router.get('/bookings', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const bookings = await db('bookings').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: bookings, page, pageSize });
  }));

  router.get('/bookings/:id', asyncHandler(async (req, res) => {
    const booking = await db('bookings').where({ id: req.params.id }).first();
    if (!booking) {
      return res.status(404).json({ error: 'booking not found' });
    }
    res.json(booking);
  }));

  router.post('/bookings', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.room_id !== 'number') {
      errors.push('room_id must be a number');
    }
    if (typeof req.body.guest_name !== 'string' || req.body.guest_name.length === 0) {
      errors.push('guest_name is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('bookings').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/bookings/:id', asyncHandler(async (req, res) => {
    const removed = await db('bookings').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'booking not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function subscriptionRouter(db) {
  const router = express.Router();

  router.get('/subscriptions', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const subscriptions = await db('subscriptions').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: subscriptions, page, pageSize });
  }));

  router.get('/subscriptions/:id', asyncHandler(async (req, res) => {
    const subscription = await db('subscriptions').where({ id: req.params.id }).first();
    if (!subscription) {
      return res.status(404).json({ error: 'subscription not found' });
    }
    res.json(subscription);
  }));

  router.post('/subscriptions', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.user_id !== 'number') {
      errors.push('user_id must be a number');
    }
    if (typeof req.body.plan !== 'string' || req.body.plan.length === 0) {
      errors.push('plan is required');
    }
    if (typeof req.body.price !== 'number') {
      errors.push('price must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('subscriptions').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/subscriptions/:id', asyncHandler(async (req, res) => {
    const removed = await db('subscriptions').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'subscription not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function notificationRouter(db) {
  const router = express.Router();

  router.get('/notifications', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const notifications = await db('notifications').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: notifications, page, pageSize });
  }));

  router.get('/notifications/:id', asyncHandler(async (req, res) => {
    const notification = await db('notifications').where({ id: req.params.id }).first();
    if (!notification) {
      return res.status(404).json({ error: 'notification not found' });
    }
    res.json(notification);
  }));

  router.post('/notifications', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.user_id !== 'number') {
      errors.push('user_id must be a number');
    }
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.message !== 'string' || req.body.message.length === 0) {
      errors.push('message is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('notifications').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/notifications/:id', asyncHandler(async (req, res) => {
    const removed = await db('notifications').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'notification not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function sessionRouter(db) {
  const router = express.Router();

  router.get('/sessions', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const sessions = await db('sessions').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: sessions, page, pageSize });
  }));

  router.get('/sessions/:id', asyncHandler(async (req, res) => {
    const session = await db('sessions').where({ id: req.params.id }).first();
    if (!session) {
      return res.status(404).json({ error: 'session not found' });
    }
    res.json(session);
  }));

  router.post('/sessions', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.user_id !== 'number') {
      errors.push('user_id must be a number');
    }
    if (typeof req.body.token !== 'string' || req.body.token.length === 0) {
      errors.push('token is required');
    }
    if (typeof req.body.ip_address !== 'string' || req.body.ip_address.length === 0) {
      errors.push('ip_address is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('sessions').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/sessions/:id', asyncHandler(async (req, res) => {
    const removed = await db('sessions').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'session not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function auditLogRouter(db) {
  const router = express.Router();

  router.get('/audit-logs', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const auditLogs = await db('audit_logs').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: auditLogs, page, pageSize });
  }));

  router.get('/audit-logs/:id', asyncHandler(async (req, res) => {
    const auditLog = await db('audit_logs').where({ id: req.params.id }).first();
    if (!auditLog) {
      return res.status(404).json({ error: 'audit log not found' });
    }
    res.json(auditLog);
  }));

  router.post('/audit-logs', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.actor_id !== 'number') {
      errors.push('actor_id must be a number');
    }
    if (typeof req.body.action !== 'string' || req.body.action.length === 0) {
      errors.push('action is required');
    }
    if (typeof req.body.target !== 'string' || req.body.target.length === 0) {
      errors.push('target is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('audit_logs').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/audit-logs/:id', asyncHandler(async (req, res) => {
    const removed = await db('audit_logs').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'audit log not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function settingRouter(db) {
  const router = express.Router();

  router.get('/settings', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const settings = await db('settings').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: settings, page, pageSize });
  }));

  router.get('/settings/:id', asyncHandler(async (req, res) => {
    const setting = await db('settings').where({ id: req.params.id }).first();
    if (!setting) {
      return res.status(404).json({ error: 'setting not found' });
    }
    res.json(setting);
  }));

  router.post('/settings', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.key !== 'string' || req.body.key.length === 0) {
      errors.push('key is required');
    }
    if (typeof req.body.value !== 'string' || req.body.value.length === 0) {
      errors.push('value is required');
    }
    if (typeof req.body.description !== 'string' || req.body.description.length === 0) {
      errors.push('description is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('settings').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/settings/:id', asyncHandler(async (req, res) => {
    const removed = await db('settings').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'setting not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function recipeRouter(db) {
  const router = express.Router();

  router.get('/recipes', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const recipes = await db('recipes').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: recipes, page, pageSize });
  }));

  router.get('/recipes/:id', asyncHandler(async (req, res) => {
    const recipe = await db('recipes').where({ id: req.params.id }).first();
    if (!recipe) {
      return res.status(404).json({ error: 'recipe not found' });
    }
    res.json(recipe);
  }));

  router.post('/recipes', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.servings !== 'number') {
      errors.push('servings must be a number');
    }
    if (typeof req.body.prep_minutes !== 'number') {
      errors.push('prep_minutes must be a number');
    }
    if (typeof req.body.cook_minutes !== 'number') {
      errors.push('cook_minutes must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('recipes').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/recipes/:id', asyncHandler(async (req, res) => {
    const removed = await db('recipes').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'recipe not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function ingredientRouter(db) {
  const router = express.Router();

  router.get('/ingredients', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const ingredients = await db('ingredients').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: ingredients, page, pageSize });
  }));

  router.get('/ingredients/:id', asyncHandler(async (req, res) => {
    const ingredient = await db('ingredients').where({ id: req.params.id }).first();
    if (!ingredient) {
      return res.status(404).json({ error: 'ingredient not found' });
    }
    res.json(ingredient);
  }));

  router.post('/ingredients', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.recipe_id !== 'number') {
      errors.push('recipe_id must be a number');
    }
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.amount !== 'number') {
      errors.push('amount must be a number');
    }
    if (typeof req.body.unit !== 'string' || req.body.unit.length === 0) {
      errors.push('unit is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('ingredients').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/ingredients/:id', asyncHandler(async (req, res) => {
    const removed = await db('ingredients').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'ingredient not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function playlistRouter(db) {
  const router = express.Router();

  router.get('/playlists', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const playlists = await db('playlists').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: playlists, page, pageSize });
  }));

  router.get('/playlists/:id', asyncHandler(async (req, res) => {
    const playlist = await db('playlists').where({ id: req.params.id }).first();
    if (!playlist) {
      return res.status(404).json({ error: 'playlist not found' });
    }
    res.json(playlist);
  }));

  router.post('/playlists', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.owner_id !== 'number') {
      errors.push('owner_id must be a number');
    }
    if (typeof req.body.track_count !== 'number') {
      errors.push('track_count must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('playlists').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/playlists/:id', asyncHandler(async (req, res) => {
    const removed = await db('playlists').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'playlist not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function trackRouter(db) {
  const router = express.Router();

  router.get('/tracks', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const tracks = await db('tracks').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: tracks, page, pageSize });
  }));

  router.get('/tracks/:id', asyncHandler(async (req, res) => {
    const track = await db('tracks').where({ id: req.params.id }).first();
    if (!track) {
      return res.status(404).json({ error: 'track not found' });
    }
    res.json(track);
  }));

  router.post('/tracks', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.title !== 'string' || req.body.title.length === 0) {
      errors.push('title is required');
    }
    if (typeof req.body.artist !== 'string' || req.body.artist.length === 0) {
      errors.push('artist is required');
    }
    if (typeof req.body.album !== 'string' || req.body.album.length === 0) {
      errors.push('album is required');
    }
    if (typeof req.body.duration_seconds !== 'number') {
      errors.push('duration_seconds must be a number');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('tracks').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/tracks/:id', asyncHandler(async (req, res) => {
    const removed = await db('tracks').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'track not found' });
    }
    res.status(204).end();
  }));

  return router;
}

function deviceRouter(db) {
  const router = express.Router();

  router.get('/devices', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const pageSize = Math.min(100, parseInt(req.query.pageSize, 10) || 20);
    const devices = await db('devices').limit(pageSize).offset((page - 1) * pageSize);
    res.json({ items: devices, page, pageSize });
  }));

  router.get('/devices/:id', asyncHandler(async (req, res) => {
    const device = await db('devices').where({ id: req.params.id }).first();
    if (!device) {
      return res.status(404).json({ error: 'device not found' });
    }
    res.json(device);
  }));

  router.post('/devices', asyncHandler(async (req, res) => {
    const errors = [];
    if (typeof req.body.name !== 'string' || req.body.name.length === 0) {
      errors.push('name is required');
    }
    if (typeof req.body.serial_number !== 'string' || req.body.serial_number.length === 0) {
      errors.push('serial_number is required');
    }
    if (typeof req.body.firmware !== 'string' || req.body.firmware.length === 0) {
      errors.push('firmware is required');
    }
    if (errors.length > 0) {
      return res.status(400).json({ errors });
    }
    const [id] = await db('devices').insert(req.body);
    res.status(201).json({ id, ...req.body });
  }));

  router.delete('/devices/:id', asyncHandler(async (req, res) => {
    const removed = await db('devices').where({ id: req.params.id }).del();
    if (!removed) {
      return res.status(404).json({ error: 'device not found' });
    }
    res.status(204).end();
  }));

  return router;
}

module.exports = {
  userRouter,
  productRouter,
  orderRouter,
  orderItemRouter,
  customerRouter,
  invoiceRouter,
  paymentRouter,
  categoryRouter,
  reviewRouter,
  addressRouter,
  shipmentRouter,
  warehouseRouter,
  supplierRouter,
  couponRouter,
  employeeRouter,
  departmentRouter,
  projectRouter,
  taskRouter,
  commentRouter,
  tagRouter,
  articleRouter,
  authorRouter,
  bookRouter,
  libraryRouter,
  loanRouter,
  memberRouter,
  eventRouter,
  ticketRouter,
  venueRouter,
  vehicleRouter,
  driverRouter,
  tripRouter,
  patientRouter,
  doctorRouter,
  appointmentRouter,
  courseRouter,
  studentRouter,
  enrollmentRouter,
  roomRouter,
  bookingRouter,
  subscriptionRouter,
  notificationRouter,
  sessionRouter,
  auditLogRouter,
  settingRouter,
  recipeRouter,
  ingredientRouter,
  playlistRouter,
  trackRouter,
  deviceRouter,
};
