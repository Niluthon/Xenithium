import React, { useEffect, useState } from 'react';

import * as api from './api';

export function UserList() {
  const [items, setItems] = useState<api.User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.usersApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Username</th>
          <th>Email</th>
          <th>Full Name</th>
          <th>Is Active</th>
          <th>Created At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.username}</td>
            <td>{item.email}</td>
            <td>{item.fullName}</td>
            <td>{item.isActive ? 'Yes' : 'No'}</td>
            <td>{item.createdAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function UserForm({ onSaved }: { onSaved: (item: api.User) => void }) {
  const [form, setForm] = useState<api.NewUser>({ username: '', email: '', fullName: '', isActive: false, createdAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.usersApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Username
        <input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
      </label>
      <label>
        Email
        <input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
      </label>
      <label>
        Full Name
        <input value={form.fullName} onChange={(e) => setForm({ ...form, fullName: e.target.value })} />
      </label>
      <label>
        Is Active
        <input type="checkbox" checked={form.isActive} onChange={(e) => setForm({ ...form, isActive: e.target.checked })} />
      </label>
      <label>
        Created At
        <input value={form.createdAt} onChange={(e) => setForm({ ...form, createdAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function ProductList() {
  const [items, setItems] = useState<api.Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.productsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Sku</th>
          <th>Name</th>
          <th>Description</th>
          <th>Price</th>
          <th>Stock</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.sku}</td>
            <td>{item.name}</td>
            <td>{item.description}</td>
            <td>{item.price}</td>
            <td>{item.stock}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ProductForm({ onSaved }: { onSaved: (item: api.Product) => void }) {
  const [form, setForm] = useState<api.NewProduct>({ sku: '', name: '', description: '', price: 0, stock: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.productsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Sku
        <input value={form.sku} onChange={(e) => setForm({ ...form, sku: e.target.value })} />
      </label>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Description
        <input value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
      </label>
      <label>
        Price
        <input type="number" value={form.price} onChange={(e) => setForm({ ...form, price: Number(e.target.value) })} />
      </label>
      <label>
        Stock
        <input type="number" value={form.stock} onChange={(e) => setForm({ ...form, stock: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function OrderList() {
  const [items, setItems] = useState<api.Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.ordersApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Customer Id</th>
          <th>Status</th>
          <th>Total</th>
          <th>Currency</th>
          <th>Placed At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.customerId}</td>
            <td>{item.status}</td>
            <td>{item.total}</td>
            <td>{item.currency}</td>
            <td>{item.placedAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function OrderForm({ onSaved }: { onSaved: (item: api.Order) => void }) {
  const [form, setForm] = useState<api.NewOrder>({ customerId: 0, status: '', total: 0, currency: '', placedAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.ordersApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Customer Id
        <input type="number" value={form.customerId} onChange={(e) => setForm({ ...form, customerId: Number(e.target.value) })} />
      </label>
      <label>
        Status
        <input value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value })} />
      </label>
      <label>
        Total
        <input type="number" value={form.total} onChange={(e) => setForm({ ...form, total: Number(e.target.value) })} />
      </label>
      <label>
        Currency
        <input value={form.currency} onChange={(e) => setForm({ ...form, currency: e.target.value })} />
      </label>
      <label>
        Placed At
        <input value={form.placedAt} onChange={(e) => setForm({ ...form, placedAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function OrderItemList() {
  const [items, setItems] = useState<api.OrderItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.orderItemsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Order Id</th>
          <th>Product Id</th>
          <th>Quantity</th>
          <th>Unit Price</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.orderId}</td>
            <td>{item.productId}</td>
            <td>{item.quantity}</td>
            <td>{item.unitPrice}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function OrderItemForm({ onSaved }: { onSaved: (item: api.OrderItem) => void }) {
  const [form, setForm] = useState<api.NewOrderItem>({ orderId: 0, productId: 0, quantity: 0, unitPrice: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.orderItemsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Order Id
        <input type="number" value={form.orderId} onChange={(e) => setForm({ ...form, orderId: Number(e.target.value) })} />
      </label>
      <label>
        Product Id
        <input type="number" value={form.productId} onChange={(e) => setForm({ ...form, productId: Number(e.target.value) })} />
      </label>
      <label>
        Quantity
        <input type="number" value={form.quantity} onChange={(e) => setForm({ ...form, quantity: Number(e.target.value) })} />
      </label>
      <label>
        Unit Price
        <input type="number" value={form.unitPrice} onChange={(e) => setForm({ ...form, unitPrice: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function CustomerList() {
  const [items, setItems] = useState<api.Customer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.customersApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>First Name</th>
          <th>Last Name</th>
          <th>Email</th>
          <th>Phone</th>
          <th>Country</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.firstName}</td>
            <td>{item.lastName}</td>
            <td>{item.email}</td>
            <td>{item.phone}</td>
            <td>{item.country}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CustomerForm({ onSaved }: { onSaved: (item: api.Customer) => void }) {
  const [form, setForm] = useState<api.NewCustomer>({ firstName: '', lastName: '', email: '', phone: '', country: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.customersApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        First Name
        <input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} />
      </label>
      <label>
        Last Name
        <input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} />
      </label>
      <label>
        Email
        <input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
      </label>
      <label>
        Phone
        <input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
      </label>
      <label>
        Country
        <input value={form.country} onChange={(e) => setForm({ ...form, country: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function InvoiceList() {
  const [items, setItems] = useState<api.Invoice[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.invoicesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Order Id</th>
          <th>Number</th>
          <th>Amount</th>
          <th>Due Date</th>
          <th>Paid</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.orderId}</td>
            <td>{item.number}</td>
            <td>{item.amount}</td>
            <td>{item.dueDate}</td>
            <td>{item.paid ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function InvoiceForm({ onSaved }: { onSaved: (item: api.Invoice) => void }) {
  const [form, setForm] = useState<api.NewInvoice>({ orderId: 0, number: '', amount: 0, dueDate: '', paid: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.invoicesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Order Id
        <input type="number" value={form.orderId} onChange={(e) => setForm({ ...form, orderId: Number(e.target.value) })} />
      </label>
      <label>
        Number
        <input value={form.number} onChange={(e) => setForm({ ...form, number: e.target.value })} />
      </label>
      <label>
        Amount
        <input type="number" value={form.amount} onChange={(e) => setForm({ ...form, amount: Number(e.target.value) })} />
      </label>
      <label>
        Due Date
        <input value={form.dueDate} onChange={(e) => setForm({ ...form, dueDate: e.target.value })} />
      </label>
      <label>
        Paid
        <input type="checkbox" checked={form.paid} onChange={(e) => setForm({ ...form, paid: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function PaymentList() {
  const [items, setItems] = useState<api.Payment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.paymentsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Invoice Id</th>
          <th>Method</th>
          <th>Amount</th>
          <th>Reference</th>
          <th>Received At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.invoiceId}</td>
            <td>{item.method}</td>
            <td>{item.amount}</td>
            <td>{item.reference}</td>
            <td>{item.receivedAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function PaymentForm({ onSaved }: { onSaved: (item: api.Payment) => void }) {
  const [form, setForm] = useState<api.NewPayment>({ invoiceId: 0, method: '', amount: 0, reference: '', receivedAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.paymentsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Invoice Id
        <input type="number" value={form.invoiceId} onChange={(e) => setForm({ ...form, invoiceId: Number(e.target.value) })} />
      </label>
      <label>
        Method
        <input value={form.method} onChange={(e) => setForm({ ...form, method: e.target.value })} />
      </label>
      <label>
        Amount
        <input type="number" value={form.amount} onChange={(e) => setForm({ ...form, amount: Number(e.target.value) })} />
      </label>
      <label>
        Reference
        <input value={form.reference} onChange={(e) => setForm({ ...form, reference: e.target.value })} />
      </label>
      <label>
        Received At
        <input value={form.receivedAt} onChange={(e) => setForm({ ...form, receivedAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function CategoryList() {
  const [items, setItems] = useState<api.Category[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.categoriesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Slug</th>
          <th>Parent Id</th>
          <th>Position</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.slug}</td>
            <td>{item.parentId}</td>
            <td>{item.position}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CategoryForm({ onSaved }: { onSaved: (item: api.Category) => void }) {
  const [form, setForm] = useState<api.NewCategory>({ name: '', slug: '', parentId: 0, position: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.categoriesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Slug
        <input value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} />
      </label>
      <label>
        Parent Id
        <input type="number" value={form.parentId} onChange={(e) => setForm({ ...form, parentId: Number(e.target.value) })} />
      </label>
      <label>
        Position
        <input type="number" value={form.position} onChange={(e) => setForm({ ...form, position: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function ReviewList() {
  const [items, setItems] = useState<api.Review[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.reviewsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Product Id</th>
          <th>User Id</th>
          <th>Rating</th>
          <th>Title</th>
          <th>Body</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.productId}</td>
            <td>{item.userId}</td>
            <td>{item.rating}</td>
            <td>{item.title}</td>
            <td>{item.body}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ReviewForm({ onSaved }: { onSaved: (item: api.Review) => void }) {
  const [form, setForm] = useState<api.NewReview>({ productId: 0, userId: 0, rating: 0, title: '', body: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.reviewsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Product Id
        <input type="number" value={form.productId} onChange={(e) => setForm({ ...form, productId: Number(e.target.value) })} />
      </label>
      <label>
        User Id
        <input type="number" value={form.userId} onChange={(e) => setForm({ ...form, userId: Number(e.target.value) })} />
      </label>
      <label>
        Rating
        <input type="number" value={form.rating} onChange={(e) => setForm({ ...form, rating: Number(e.target.value) })} />
      </label>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Body
        <input value={form.body} onChange={(e) => setForm({ ...form, body: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function AddressList() {
  const [items, setItems] = useState<api.Address[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.addressesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Customer Id</th>
          <th>Street</th>
          <th>City</th>
          <th>Postal Code</th>
          <th>Country</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.customerId}</td>
            <td>{item.street}</td>
            <td>{item.city}</td>
            <td>{item.postalCode}</td>
            <td>{item.country}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AddressForm({ onSaved }: { onSaved: (item: api.Address) => void }) {
  const [form, setForm] = useState<api.NewAddress>({ customerId: 0, street: '', city: '', postalCode: '', country: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.addressesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Customer Id
        <input type="number" value={form.customerId} onChange={(e) => setForm({ ...form, customerId: Number(e.target.value) })} />
      </label>
      <label>
        Street
        <input value={form.street} onChange={(e) => setForm({ ...form, street: e.target.value })} />
      </label>
      <label>
        City
        <input value={form.city} onChange={(e) => setForm({ ...form, city: e.target.value })} />
      </label>
      <label>
        Postal Code
        <input value={form.postalCode} onChange={(e) => setForm({ ...form, postalCode: e.target.value })} />
      </label>
      <label>
        Country
        <input value={form.country} onChange={(e) => setForm({ ...form, country: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function ShipmentList() {
  const [items, setItems] = useState<api.Shipment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.shipmentsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Order Id</th>
          <th>Carrier</th>
          <th>Tracking Number</th>
          <th>Shipped At</th>
          <th>Delivered</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.orderId}</td>
            <td>{item.carrier}</td>
            <td>{item.trackingNumber}</td>
            <td>{item.shippedAt}</td>
            <td>{item.delivered ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ShipmentForm({ onSaved }: { onSaved: (item: api.Shipment) => void }) {
  const [form, setForm] = useState<api.NewShipment>({ orderId: 0, carrier: '', trackingNumber: '', shippedAt: '', delivered: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.shipmentsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Order Id
        <input type="number" value={form.orderId} onChange={(e) => setForm({ ...form, orderId: Number(e.target.value) })} />
      </label>
      <label>
        Carrier
        <input value={form.carrier} onChange={(e) => setForm({ ...form, carrier: e.target.value })} />
      </label>
      <label>
        Tracking Number
        <input value={form.trackingNumber} onChange={(e) => setForm({ ...form, trackingNumber: e.target.value })} />
      </label>
      <label>
        Shipped At
        <input value={form.shippedAt} onChange={(e) => setForm({ ...form, shippedAt: e.target.value })} />
      </label>
      <label>
        Delivered
        <input type="checkbox" checked={form.delivered} onChange={(e) => setForm({ ...form, delivered: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function WarehouseList() {
  const [items, setItems] = useState<api.Warehouse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.warehousesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>City</th>
          <th>Capacity</th>
          <th>Is Open</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.city}</td>
            <td>{item.capacity}</td>
            <td>{item.isOpen ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function WarehouseForm({ onSaved }: { onSaved: (item: api.Warehouse) => void }) {
  const [form, setForm] = useState<api.NewWarehouse>({ name: '', city: '', capacity: 0, isOpen: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.warehousesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        City
        <input value={form.city} onChange={(e) => setForm({ ...form, city: e.target.value })} />
      </label>
      <label>
        Capacity
        <input type="number" value={form.capacity} onChange={(e) => setForm({ ...form, capacity: Number(e.target.value) })} />
      </label>
      <label>
        Is Open
        <input type="checkbox" checked={form.isOpen} onChange={(e) => setForm({ ...form, isOpen: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function SupplierList() {
  const [items, setItems] = useState<api.Supplier[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.suppliersApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Contact Email</th>
          <th>Phone</th>
          <th>Country</th>
          <th>Rating</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.contactEmail}</td>
            <td>{item.phone}</td>
            <td>{item.country}</td>
            <td>{item.rating}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SupplierForm({ onSaved }: { onSaved: (item: api.Supplier) => void }) {
  const [form, setForm] = useState<api.NewSupplier>({ name: '', contactEmail: '', phone: '', country: '', rating: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.suppliersApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Contact Email
        <input value={form.contactEmail} onChange={(e) => setForm({ ...form, contactEmail: e.target.value })} />
      </label>
      <label>
        Phone
        <input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
      </label>
      <label>
        Country
        <input value={form.country} onChange={(e) => setForm({ ...form, country: e.target.value })} />
      </label>
      <label>
        Rating
        <input type="number" value={form.rating} onChange={(e) => setForm({ ...form, rating: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function CouponList() {
  const [items, setItems] = useState<api.Coupon[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.couponsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Code</th>
          <th>Discount Percent</th>
          <th>Valid Until</th>
          <th>Max Uses</th>
          <th>Is Active</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.code}</td>
            <td>{item.discountPercent}</td>
            <td>{item.validUntil}</td>
            <td>{item.maxUses}</td>
            <td>{item.isActive ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CouponForm({ onSaved }: { onSaved: (item: api.Coupon) => void }) {
  const [form, setForm] = useState<api.NewCoupon>({ code: '', discountPercent: 0, validUntil: '', maxUses: 0, isActive: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.couponsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Code
        <input value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} />
      </label>
      <label>
        Discount Percent
        <input type="number" value={form.discountPercent} onChange={(e) => setForm({ ...form, discountPercent: Number(e.target.value) })} />
      </label>
      <label>
        Valid Until
        <input value={form.validUntil} onChange={(e) => setForm({ ...form, validUntil: e.target.value })} />
      </label>
      <label>
        Max Uses
        <input type="number" value={form.maxUses} onChange={(e) => setForm({ ...form, maxUses: Number(e.target.value) })} />
      </label>
      <label>
        Is Active
        <input type="checkbox" checked={form.isActive} onChange={(e) => setForm({ ...form, isActive: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function EmployeeList() {
  const [items, setItems] = useState<api.Employee[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.employeesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>First Name</th>
          <th>Last Name</th>
          <th>Email</th>
          <th>Department</th>
          <th>Salary</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.firstName}</td>
            <td>{item.lastName}</td>
            <td>{item.email}</td>
            <td>{item.department}</td>
            <td>{item.salary}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function EmployeeForm({ onSaved }: { onSaved: (item: api.Employee) => void }) {
  const [form, setForm] = useState<api.NewEmployee>({ firstName: '', lastName: '', email: '', department: '', salary: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.employeesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        First Name
        <input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} />
      </label>
      <label>
        Last Name
        <input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} />
      </label>
      <label>
        Email
        <input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
      </label>
      <label>
        Department
        <input value={form.department} onChange={(e) => setForm({ ...form, department: e.target.value })} />
      </label>
      <label>
        Salary
        <input type="number" value={form.salary} onChange={(e) => setForm({ ...form, salary: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function DepartmentList() {
  const [items, setItems] = useState<api.Department[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.departmentsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Code</th>
          <th>Manager Id</th>
          <th>Budget</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.code}</td>
            <td>{item.managerId}</td>
            <td>{item.budget}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DepartmentForm({ onSaved }: { onSaved: (item: api.Department) => void }) {
  const [form, setForm] = useState<api.NewDepartment>({ name: '', code: '', managerId: 0, budget: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.departmentsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Code
        <input value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} />
      </label>
      <label>
        Manager Id
        <input type="number" value={form.managerId} onChange={(e) => setForm({ ...form, managerId: Number(e.target.value) })} />
      </label>
      <label>
        Budget
        <input type="number" value={form.budget} onChange={(e) => setForm({ ...form, budget: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function ProjectList() {
  const [items, setItems] = useState<api.Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.projectsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Description</th>
          <th>Start Date</th>
          <th>End Date</th>
          <th>Budget</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.description}</td>
            <td>{item.startDate}</td>
            <td>{item.endDate}</td>
            <td>{item.budget}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ProjectForm({ onSaved }: { onSaved: (item: api.Project) => void }) {
  const [form, setForm] = useState<api.NewProject>({ name: '', description: '', startDate: '', endDate: '', budget: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.projectsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Description
        <input value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
      </label>
      <label>
        Start Date
        <input value={form.startDate} onChange={(e) => setForm({ ...form, startDate: e.target.value })} />
      </label>
      <label>
        End Date
        <input value={form.endDate} onChange={(e) => setForm({ ...form, endDate: e.target.value })} />
      </label>
      <label>
        Budget
        <input type="number" value={form.budget} onChange={(e) => setForm({ ...form, budget: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function TaskList() {
  const [items, setItems] = useState<api.Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tasksApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Project Id</th>
          <th>Title</th>
          <th>Status</th>
          <th>Priority</th>
          <th>Due Date</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.projectId}</td>
            <td>{item.title}</td>
            <td>{item.status}</td>
            <td>{item.priority}</td>
            <td>{item.dueDate}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TaskForm({ onSaved }: { onSaved: (item: api.Task) => void }) {
  const [form, setForm] = useState<api.NewTask>({ projectId: 0, title: '', status: '', priority: 0, dueDate: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.tasksApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Project Id
        <input type="number" value={form.projectId} onChange={(e) => setForm({ ...form, projectId: Number(e.target.value) })} />
      </label>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Status
        <input value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value })} />
      </label>
      <label>
        Priority
        <input type="number" value={form.priority} onChange={(e) => setForm({ ...form, priority: Number(e.target.value) })} />
      </label>
      <label>
        Due Date
        <input value={form.dueDate} onChange={(e) => setForm({ ...form, dueDate: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function CommentList() {
  const [items, setItems] = useState<api.Comment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.commentsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Task Id</th>
          <th>Author Id</th>
          <th>Body</th>
          <th>Created At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.taskId}</td>
            <td>{item.authorId}</td>
            <td>{item.body}</td>
            <td>{item.createdAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CommentForm({ onSaved }: { onSaved: (item: api.Comment) => void }) {
  const [form, setForm] = useState<api.NewComment>({ taskId: 0, authorId: 0, body: '', createdAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.commentsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Task Id
        <input type="number" value={form.taskId} onChange={(e) => setForm({ ...form, taskId: Number(e.target.value) })} />
      </label>
      <label>
        Author Id
        <input type="number" value={form.authorId} onChange={(e) => setForm({ ...form, authorId: Number(e.target.value) })} />
      </label>
      <label>
        Body
        <input value={form.body} onChange={(e) => setForm({ ...form, body: e.target.value })} />
      </label>
      <label>
        Created At
        <input value={form.createdAt} onChange={(e) => setForm({ ...form, createdAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function TagList() {
  const [items, setItems] = useState<api.Tag[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tagsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Color</th>
          <th>Usage Count</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.color}</td>
            <td>{item.usageCount}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TagForm({ onSaved }: { onSaved: (item: api.Tag) => void }) {
  const [form, setForm] = useState<api.NewTag>({ name: '', color: '', usageCount: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.tagsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Color
        <input value={form.color} onChange={(e) => setForm({ ...form, color: e.target.value })} />
      </label>
      <label>
        Usage Count
        <input type="number" value={form.usageCount} onChange={(e) => setForm({ ...form, usageCount: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function ArticleList() {
  const [items, setItems] = useState<api.Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.articlesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Slug</th>
          <th>Body</th>
          <th>Author Id</th>
          <th>Published</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.title}</td>
            <td>{item.slug}</td>
            <td>{item.body}</td>
            <td>{item.authorId}</td>
            <td>{item.published ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ArticleForm({ onSaved }: { onSaved: (item: api.Article) => void }) {
  const [form, setForm] = useState<api.NewArticle>({ title: '', slug: '', body: '', authorId: 0, published: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.articlesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Slug
        <input value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} />
      </label>
      <label>
        Body
        <input value={form.body} onChange={(e) => setForm({ ...form, body: e.target.value })} />
      </label>
      <label>
        Author Id
        <input type="number" value={form.authorId} onChange={(e) => setForm({ ...form, authorId: Number(e.target.value) })} />
      </label>
      <label>
        Published
        <input type="checkbox" checked={form.published} onChange={(e) => setForm({ ...form, published: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function AuthorList() {
  const [items, setItems] = useState<api.Author[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.authorsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Bio</th>
          <th>Email</th>
          <th>Website</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.bio}</td>
            <td>{item.email}</td>
            <td>{item.website}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AuthorForm({ onSaved }: { onSaved: (item: api.Author) => void }) {
  const [form, setForm] = useState<api.NewAuthor>({ name: '', bio: '', email: '', website: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.authorsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Bio
        <input value={form.bio} onChange={(e) => setForm({ ...form, bio: e.target.value })} />
      </label>
      <label>
        Email
        <input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
      </label>
      <label>
        Website
        <input value={form.website} onChange={(e) => setForm({ ...form, website: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function BookList() {
  const [items, setItems] = useState<api.Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.booksApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Isbn</th>
          <th>Author Id</th>
          <th>Year</th>
          <th>Pages</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.title}</td>
            <td>{item.isbn}</td>
            <td>{item.authorId}</td>
            <td>{item.year}</td>
            <td>{item.pages}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BookForm({ onSaved }: { onSaved: (item: api.Book) => void }) {
  const [form, setForm] = useState<api.NewBook>({ title: '', isbn: '', authorId: 0, year: 0, pages: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.booksApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Isbn
        <input value={form.isbn} onChange={(e) => setForm({ ...form, isbn: e.target.value })} />
      </label>
      <label>
        Author Id
        <input type="number" value={form.authorId} onChange={(e) => setForm({ ...form, authorId: Number(e.target.value) })} />
      </label>
      <label>
        Year
        <input type="number" value={form.year} onChange={(e) => setForm({ ...form, year: Number(e.target.value) })} />
      </label>
      <label>
        Pages
        <input type="number" value={form.pages} onChange={(e) => setForm({ ...form, pages: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function LibraryList() {
  const [items, setItems] = useState<api.Library[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.librariesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>City</th>
          <th>Opening Hour</th>
          <th>Closing Hour</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.city}</td>
            <td>{item.openingHour}</td>
            <td>{item.closingHour}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function LibraryForm({ onSaved }: { onSaved: (item: api.Library) => void }) {
  const [form, setForm] = useState<api.NewLibrary>({ name: '', city: '', openingHour: 0, closingHour: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.librariesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        City
        <input value={form.city} onChange={(e) => setForm({ ...form, city: e.target.value })} />
      </label>
      <label>
        Opening Hour
        <input type="number" value={form.openingHour} onChange={(e) => setForm({ ...form, openingHour: Number(e.target.value) })} />
      </label>
      <label>
        Closing Hour
        <input type="number" value={form.closingHour} onChange={(e) => setForm({ ...form, closingHour: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function LoanList() {
  const [items, setItems] = useState<api.Loan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.loansApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Book Id</th>
          <th>Member Id</th>
          <th>Borrowed At</th>
          <th>Due Date</th>
          <th>Returned</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.bookId}</td>
            <td>{item.memberId}</td>
            <td>{item.borrowedAt}</td>
            <td>{item.dueDate}</td>
            <td>{item.returned ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function LoanForm({ onSaved }: { onSaved: (item: api.Loan) => void }) {
  const [form, setForm] = useState<api.NewLoan>({ bookId: 0, memberId: 0, borrowedAt: '', dueDate: '', returned: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.loansApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Book Id
        <input type="number" value={form.bookId} onChange={(e) => setForm({ ...form, bookId: Number(e.target.value) })} />
      </label>
      <label>
        Member Id
        <input type="number" value={form.memberId} onChange={(e) => setForm({ ...form, memberId: Number(e.target.value) })} />
      </label>
      <label>
        Borrowed At
        <input value={form.borrowedAt} onChange={(e) => setForm({ ...form, borrowedAt: e.target.value })} />
      </label>
      <label>
        Due Date
        <input value={form.dueDate} onChange={(e) => setForm({ ...form, dueDate: e.target.value })} />
      </label>
      <label>
        Returned
        <input type="checkbox" checked={form.returned} onChange={(e) => setForm({ ...form, returned: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function MemberList() {
  const [items, setItems] = useState<api.Member[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.membersApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>First Name</th>
          <th>Last Name</th>
          <th>Email</th>
          <th>Joined At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.firstName}</td>
            <td>{item.lastName}</td>
            <td>{item.email}</td>
            <td>{item.joinedAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function MemberForm({ onSaved }: { onSaved: (item: api.Member) => void }) {
  const [form, setForm] = useState<api.NewMember>({ firstName: '', lastName: '', email: '', joinedAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.membersApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        First Name
        <input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} />
      </label>
      <label>
        Last Name
        <input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} />
      </label>
      <label>
        Email
        <input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
      </label>
      <label>
        Joined At
        <input value={form.joinedAt} onChange={(e) => setForm({ ...form, joinedAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function EventList() {
  const [items, setItems] = useState<api.Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.eventsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Location</th>
          <th>Starts At</th>
          <th>Ends At</th>
          <th>Capacity</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.title}</td>
            <td>{item.location}</td>
            <td>{item.startsAt}</td>
            <td>{item.endsAt}</td>
            <td>{item.capacity}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function EventForm({ onSaved }: { onSaved: (item: api.Event) => void }) {
  const [form, setForm] = useState<api.NewEvent>({ title: '', location: '', startsAt: '', endsAt: '', capacity: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.eventsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Location
        <input value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} />
      </label>
      <label>
        Starts At
        <input value={form.startsAt} onChange={(e) => setForm({ ...form, startsAt: e.target.value })} />
      </label>
      <label>
        Ends At
        <input value={form.endsAt} onChange={(e) => setForm({ ...form, endsAt: e.target.value })} />
      </label>
      <label>
        Capacity
        <input type="number" value={form.capacity} onChange={(e) => setForm({ ...form, capacity: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function TicketList() {
  const [items, setItems] = useState<api.Ticket[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.ticketsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Event Id</th>
          <th>Holder Name</th>
          <th>Seat</th>
          <th>Price</th>
          <th>Checked In</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.eventId}</td>
            <td>{item.holderName}</td>
            <td>{item.seat}</td>
            <td>{item.price}</td>
            <td>{item.checkedIn ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TicketForm({ onSaved }: { onSaved: (item: api.Ticket) => void }) {
  const [form, setForm] = useState<api.NewTicket>({ eventId: 0, holderName: '', seat: '', price: 0, checkedIn: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.ticketsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Event Id
        <input type="number" value={form.eventId} onChange={(e) => setForm({ ...form, eventId: Number(e.target.value) })} />
      </label>
      <label>
        Holder Name
        <input value={form.holderName} onChange={(e) => setForm({ ...form, holderName: e.target.value })} />
      </label>
      <label>
        Seat
        <input value={form.seat} onChange={(e) => setForm({ ...form, seat: e.target.value })} />
      </label>
      <label>
        Price
        <input type="number" value={form.price} onChange={(e) => setForm({ ...form, price: Number(e.target.value) })} />
      </label>
      <label>
        Checked In
        <input type="checkbox" checked={form.checkedIn} onChange={(e) => setForm({ ...form, checkedIn: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function VenueList() {
  const [items, setItems] = useState<api.Venue[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.venuesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Address</th>
          <th>City</th>
          <th>Capacity</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.address}</td>
            <td>{item.city}</td>
            <td>{item.capacity}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function VenueForm({ onSaved }: { onSaved: (item: api.Venue) => void }) {
  const [form, setForm] = useState<api.NewVenue>({ name: '', address: '', city: '', capacity: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.venuesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Address
        <input value={form.address} onChange={(e) => setForm({ ...form, address: e.target.value })} />
      </label>
      <label>
        City
        <input value={form.city} onChange={(e) => setForm({ ...form, city: e.target.value })} />
      </label>
      <label>
        Capacity
        <input type="number" value={form.capacity} onChange={(e) => setForm({ ...form, capacity: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function VehicleList() {
  const [items, setItems] = useState<api.Vehicle[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.vehiclesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Plate</th>
          <th>Make</th>
          <th>Model</th>
          <th>Year</th>
          <th>Mileage</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.plate}</td>
            <td>{item.make}</td>
            <td>{item.model}</td>
            <td>{item.year}</td>
            <td>{item.mileage}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function VehicleForm({ onSaved }: { onSaved: (item: api.Vehicle) => void }) {
  const [form, setForm] = useState<api.NewVehicle>({ plate: '', make: '', model: '', year: 0, mileage: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.vehiclesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Plate
        <input value={form.plate} onChange={(e) => setForm({ ...form, plate: e.target.value })} />
      </label>
      <label>
        Make
        <input value={form.make} onChange={(e) => setForm({ ...form, make: e.target.value })} />
      </label>
      <label>
        Model
        <input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} />
      </label>
      <label>
        Year
        <input type="number" value={form.year} onChange={(e) => setForm({ ...form, year: Number(e.target.value) })} />
      </label>
      <label>
        Mileage
        <input type="number" value={form.mileage} onChange={(e) => setForm({ ...form, mileage: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function DriverList() {
  const [items, setItems] = useState<api.Driver[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.driversApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>First Name</th>
          <th>Last Name</th>
          <th>License Number</th>
          <th>Rating</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.firstName}</td>
            <td>{item.lastName}</td>
            <td>{item.licenseNumber}</td>
            <td>{item.rating}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DriverForm({ onSaved }: { onSaved: (item: api.Driver) => void }) {
  const [form, setForm] = useState<api.NewDriver>({ firstName: '', lastName: '', licenseNumber: '', rating: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.driversApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        First Name
        <input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} />
      </label>
      <label>
        Last Name
        <input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} />
      </label>
      <label>
        License Number
        <input value={form.licenseNumber} onChange={(e) => setForm({ ...form, licenseNumber: e.target.value })} />
      </label>
      <label>
        Rating
        <input type="number" value={form.rating} onChange={(e) => setForm({ ...form, rating: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function TripList() {
  const [items, setItems] = useState<api.Trip[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tripsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Vehicle Id</th>
          <th>Driver Id</th>
          <th>Distance Km</th>
          <th>Started At</th>
          <th>Fare</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.vehicleId}</td>
            <td>{item.driverId}</td>
            <td>{item.distanceKm}</td>
            <td>{item.startedAt}</td>
            <td>{item.fare}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TripForm({ onSaved }: { onSaved: (item: api.Trip) => void }) {
  const [form, setForm] = useState<api.NewTrip>({ vehicleId: 0, driverId: 0, distanceKm: 0, startedAt: '', fare: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.tripsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Vehicle Id
        <input type="number" value={form.vehicleId} onChange={(e) => setForm({ ...form, vehicleId: Number(e.target.value) })} />
      </label>
      <label>
        Driver Id
        <input type="number" value={form.driverId} onChange={(e) => setForm({ ...form, driverId: Number(e.target.value) })} />
      </label>
      <label>
        Distance Km
        <input type="number" value={form.distanceKm} onChange={(e) => setForm({ ...form, distanceKm: Number(e.target.value) })} />
      </label>
      <label>
        Started At
        <input value={form.startedAt} onChange={(e) => setForm({ ...form, startedAt: e.target.value })} />
      </label>
      <label>
        Fare
        <input type="number" value={form.fare} onChange={(e) => setForm({ ...form, fare: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function PatientList() {
  const [items, setItems] = useState<api.Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.patientsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>First Name</th>
          <th>Last Name</th>
          <th>Birth Date</th>
          <th>Phone</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.firstName}</td>
            <td>{item.lastName}</td>
            <td>{item.birthDate}</td>
            <td>{item.phone}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function PatientForm({ onSaved }: { onSaved: (item: api.Patient) => void }) {
  const [form, setForm] = useState<api.NewPatient>({ firstName: '', lastName: '', birthDate: '', phone: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.patientsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        First Name
        <input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} />
      </label>
      <label>
        Last Name
        <input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} />
      </label>
      <label>
        Birth Date
        <input value={form.birthDate} onChange={(e) => setForm({ ...form, birthDate: e.target.value })} />
      </label>
      <label>
        Phone
        <input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function DoctorList() {
  const [items, setItems] = useState<api.Doctor[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.doctorsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>First Name</th>
          <th>Last Name</th>
          <th>Specialty</th>
          <th>Room</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.firstName}</td>
            <td>{item.lastName}</td>
            <td>{item.specialty}</td>
            <td>{item.room}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DoctorForm({ onSaved }: { onSaved: (item: api.Doctor) => void }) {
  const [form, setForm] = useState<api.NewDoctor>({ firstName: '', lastName: '', specialty: '', room: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.doctorsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        First Name
        <input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} />
      </label>
      <label>
        Last Name
        <input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} />
      </label>
      <label>
        Specialty
        <input value={form.specialty} onChange={(e) => setForm({ ...form, specialty: e.target.value })} />
      </label>
      <label>
        Room
        <input value={form.room} onChange={(e) => setForm({ ...form, room: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function AppointmentList() {
  const [items, setItems] = useState<api.Appointment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.appointmentsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Patient Id</th>
          <th>Doctor Id</th>
          <th>Scheduled At</th>
          <th>Reason</th>
          <th>Confirmed</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.patientId}</td>
            <td>{item.doctorId}</td>
            <td>{item.scheduledAt}</td>
            <td>{item.reason}</td>
            <td>{item.confirmed ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AppointmentForm({ onSaved }: { onSaved: (item: api.Appointment) => void }) {
  const [form, setForm] = useState<api.NewAppointment>({ patientId: 0, doctorId: 0, scheduledAt: '', reason: '', confirmed: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.appointmentsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Patient Id
        <input type="number" value={form.patientId} onChange={(e) => setForm({ ...form, patientId: Number(e.target.value) })} />
      </label>
      <label>
        Doctor Id
        <input type="number" value={form.doctorId} onChange={(e) => setForm({ ...form, doctorId: Number(e.target.value) })} />
      </label>
      <label>
        Scheduled At
        <input value={form.scheduledAt} onChange={(e) => setForm({ ...form, scheduledAt: e.target.value })} />
      </label>
      <label>
        Reason
        <input value={form.reason} onChange={(e) => setForm({ ...form, reason: e.target.value })} />
      </label>
      <label>
        Confirmed
        <input type="checkbox" checked={form.confirmed} onChange={(e) => setForm({ ...form, confirmed: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function CourseList() {
  const [items, setItems] = useState<api.Course[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.coursesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Code</th>
          <th>Credits</th>
          <th>Teacher Id</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.title}</td>
            <td>{item.code}</td>
            <td>{item.credits}</td>
            <td>{item.teacherId}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CourseForm({ onSaved }: { onSaved: (item: api.Course) => void }) {
  const [form, setForm] = useState<api.NewCourse>({ title: '', code: '', credits: 0, teacherId: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.coursesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Code
        <input value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} />
      </label>
      <label>
        Credits
        <input type="number" value={form.credits} onChange={(e) => setForm({ ...form, credits: Number(e.target.value) })} />
      </label>
      <label>
        Teacher Id
        <input type="number" value={form.teacherId} onChange={(e) => setForm({ ...form, teacherId: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function StudentList() {
  const [items, setItems] = useState<api.Student[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.studentsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>First Name</th>
          <th>Last Name</th>
          <th>Email</th>
          <th>Enrolled At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.firstName}</td>
            <td>{item.lastName}</td>
            <td>{item.email}</td>
            <td>{item.enrolledAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function StudentForm({ onSaved }: { onSaved: (item: api.Student) => void }) {
  const [form, setForm] = useState<api.NewStudent>({ firstName: '', lastName: '', email: '', enrolledAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.studentsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        First Name
        <input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} />
      </label>
      <label>
        Last Name
        <input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} />
      </label>
      <label>
        Email
        <input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
      </label>
      <label>
        Enrolled At
        <input value={form.enrolledAt} onChange={(e) => setForm({ ...form, enrolledAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function EnrollmentList() {
  const [items, setItems] = useState<api.Enrollment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.enrollmentsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Student Id</th>
          <th>Course Id</th>
          <th>Grade</th>
          <th>Completed</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.studentId}</td>
            <td>{item.courseId}</td>
            <td>{item.grade}</td>
            <td>{item.completed ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function EnrollmentForm({ onSaved }: { onSaved: (item: api.Enrollment) => void }) {
  const [form, setForm] = useState<api.NewEnrollment>({ studentId: 0, courseId: 0, grade: 0, completed: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.enrollmentsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Student Id
        <input type="number" value={form.studentId} onChange={(e) => setForm({ ...form, studentId: Number(e.target.value) })} />
      </label>
      <label>
        Course Id
        <input type="number" value={form.courseId} onChange={(e) => setForm({ ...form, courseId: Number(e.target.value) })} />
      </label>
      <label>
        Grade
        <input type="number" value={form.grade} onChange={(e) => setForm({ ...form, grade: Number(e.target.value) })} />
      </label>
      <label>
        Completed
        <input type="checkbox" checked={form.completed} onChange={(e) => setForm({ ...form, completed: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function RoomList() {
  const [items, setItems] = useState<api.Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.roomsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Number</th>
          <th>Floor</th>
          <th>Beds</th>
          <th>Price Per Night</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.number}</td>
            <td>{item.floor}</td>
            <td>{item.beds}</td>
            <td>{item.pricePerNight}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function RoomForm({ onSaved }: { onSaved: (item: api.Room) => void }) {
  const [form, setForm] = useState<api.NewRoom>({ number: '', floor: 0, beds: 0, pricePerNight: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.roomsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Number
        <input value={form.number} onChange={(e) => setForm({ ...form, number: e.target.value })} />
      </label>
      <label>
        Floor
        <input type="number" value={form.floor} onChange={(e) => setForm({ ...form, floor: Number(e.target.value) })} />
      </label>
      <label>
        Beds
        <input type="number" value={form.beds} onChange={(e) => setForm({ ...form, beds: Number(e.target.value) })} />
      </label>
      <label>
        Price Per Night
        <input type="number" value={form.pricePerNight} onChange={(e) => setForm({ ...form, pricePerNight: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function BookingList() {
  const [items, setItems] = useState<api.Booking[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.bookingsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Room Id</th>
          <th>Guest Name</th>
          <th>Check In</th>
          <th>Check Out</th>
          <th>Paid</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.roomId}</td>
            <td>{item.guestName}</td>
            <td>{item.checkIn}</td>
            <td>{item.checkOut}</td>
            <td>{item.paid ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BookingForm({ onSaved }: { onSaved: (item: api.Booking) => void }) {
  const [form, setForm] = useState<api.NewBooking>({ roomId: 0, guestName: '', checkIn: '', checkOut: '', paid: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.bookingsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Room Id
        <input type="number" value={form.roomId} onChange={(e) => setForm({ ...form, roomId: Number(e.target.value) })} />
      </label>
      <label>
        Guest Name
        <input value={form.guestName} onChange={(e) => setForm({ ...form, guestName: e.target.value })} />
      </label>
      <label>
        Check In
        <input value={form.checkIn} onChange={(e) => setForm({ ...form, checkIn: e.target.value })} />
      </label>
      <label>
        Check Out
        <input value={form.checkOut} onChange={(e) => setForm({ ...form, checkOut: e.target.value })} />
      </label>
      <label>
        Paid
        <input type="checkbox" checked={form.paid} onChange={(e) => setForm({ ...form, paid: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function SubscriptionList() {
  const [items, setItems] = useState<api.Subscription[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.subscriptionsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>User Id</th>
          <th>Plan</th>
          <th>Price</th>
          <th>Renews At</th>
          <th>Is Active</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.userId}</td>
            <td>{item.plan}</td>
            <td>{item.price}</td>
            <td>{item.renewsAt}</td>
            <td>{item.isActive ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SubscriptionForm({ onSaved }: { onSaved: (item: api.Subscription) => void }) {
  const [form, setForm] = useState<api.NewSubscription>({ userId: 0, plan: '', price: 0, renewsAt: '', isActive: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.subscriptionsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        User Id
        <input type="number" value={form.userId} onChange={(e) => setForm({ ...form, userId: Number(e.target.value) })} />
      </label>
      <label>
        Plan
        <input value={form.plan} onChange={(e) => setForm({ ...form, plan: e.target.value })} />
      </label>
      <label>
        Price
        <input type="number" value={form.price} onChange={(e) => setForm({ ...form, price: Number(e.target.value) })} />
      </label>
      <label>
        Renews At
        <input value={form.renewsAt} onChange={(e) => setForm({ ...form, renewsAt: e.target.value })} />
      </label>
      <label>
        Is Active
        <input type="checkbox" checked={form.isActive} onChange={(e) => setForm({ ...form, isActive: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function NotificationList() {
  const [items, setItems] = useState<api.Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.notificationsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>User Id</th>
          <th>Title</th>
          <th>Message</th>
          <th>Read</th>
          <th>Sent At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.userId}</td>
            <td>{item.title}</td>
            <td>{item.message}</td>
            <td>{item.read ? 'Yes' : 'No'}</td>
            <td>{item.sentAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function NotificationForm({ onSaved }: { onSaved: (item: api.Notification) => void }) {
  const [form, setForm] = useState<api.NewNotification>({ userId: 0, title: '', message: '', read: false, sentAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.notificationsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        User Id
        <input type="number" value={form.userId} onChange={(e) => setForm({ ...form, userId: Number(e.target.value) })} />
      </label>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Message
        <input value={form.message} onChange={(e) => setForm({ ...form, message: e.target.value })} />
      </label>
      <label>
        Read
        <input type="checkbox" checked={form.read} onChange={(e) => setForm({ ...form, read: e.target.checked })} />
      </label>
      <label>
        Sent At
        <input value={form.sentAt} onChange={(e) => setForm({ ...form, sentAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function SessionList() {
  const [items, setItems] = useState<api.Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.sessionsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>User Id</th>
          <th>Token</th>
          <th>Expires At</th>
          <th>Ip Address</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.userId}</td>
            <td>{item.token}</td>
            <td>{item.expiresAt}</td>
            <td>{item.ipAddress}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SessionForm({ onSaved }: { onSaved: (item: api.Session) => void }) {
  const [form, setForm] = useState<api.NewSession>({ userId: 0, token: '', expiresAt: '', ipAddress: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.sessionsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        User Id
        <input type="number" value={form.userId} onChange={(e) => setForm({ ...form, userId: Number(e.target.value) })} />
      </label>
      <label>
        Token
        <input value={form.token} onChange={(e) => setForm({ ...form, token: e.target.value })} />
      </label>
      <label>
        Expires At
        <input value={form.expiresAt} onChange={(e) => setForm({ ...form, expiresAt: e.target.value })} />
      </label>
      <label>
        Ip Address
        <input value={form.ipAddress} onChange={(e) => setForm({ ...form, ipAddress: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function AuditLogList() {
  const [items, setItems] = useState<api.AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.auditLogsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Actor Id</th>
          <th>Action</th>
          <th>Target</th>
          <th>Created At</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.actorId}</td>
            <td>{item.action}</td>
            <td>{item.target}</td>
            <td>{item.createdAt}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AuditLogForm({ onSaved }: { onSaved: (item: api.AuditLog) => void }) {
  const [form, setForm] = useState<api.NewAuditLog>({ actorId: 0, action: '', target: '', createdAt: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.auditLogsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Actor Id
        <input type="number" value={form.actorId} onChange={(e) => setForm({ ...form, actorId: Number(e.target.value) })} />
      </label>
      <label>
        Action
        <input value={form.action} onChange={(e) => setForm({ ...form, action: e.target.value })} />
      </label>
      <label>
        Target
        <input value={form.target} onChange={(e) => setForm({ ...form, target: e.target.value })} />
      </label>
      <label>
        Created At
        <input value={form.createdAt} onChange={(e) => setForm({ ...form, createdAt: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function SettingList() {
  const [items, setItems] = useState<api.Setting[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.settingsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Key</th>
          <th>Value</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.key}</td>
            <td>{item.value}</td>
            <td>{item.description}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SettingForm({ onSaved }: { onSaved: (item: api.Setting) => void }) {
  const [form, setForm] = useState<api.NewSetting>({ key: '', value: '', description: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.settingsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Key
        <input value={form.key} onChange={(e) => setForm({ ...form, key: e.target.value })} />
      </label>
      <label>
        Value
        <input value={form.value} onChange={(e) => setForm({ ...form, value: e.target.value })} />
      </label>
      <label>
        Description
        <input value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function RecipeList() {
  const [items, setItems] = useState<api.Recipe[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.recipesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Servings</th>
          <th>Prep Minutes</th>
          <th>Cook Minutes</th>
          <th>Vegetarian</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.title}</td>
            <td>{item.servings}</td>
            <td>{item.prepMinutes}</td>
            <td>{item.cookMinutes}</td>
            <td>{item.vegetarian ? 'Yes' : 'No'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function RecipeForm({ onSaved }: { onSaved: (item: api.Recipe) => void }) {
  const [form, setForm] = useState<api.NewRecipe>({ title: '', servings: 0, prepMinutes: 0, cookMinutes: 0, vegetarian: false });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.recipesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Servings
        <input type="number" value={form.servings} onChange={(e) => setForm({ ...form, servings: Number(e.target.value) })} />
      </label>
      <label>
        Prep Minutes
        <input type="number" value={form.prepMinutes} onChange={(e) => setForm({ ...form, prepMinutes: Number(e.target.value) })} />
      </label>
      <label>
        Cook Minutes
        <input type="number" value={form.cookMinutes} onChange={(e) => setForm({ ...form, cookMinutes: Number(e.target.value) })} />
      </label>
      <label>
        Vegetarian
        <input type="checkbox" checked={form.vegetarian} onChange={(e) => setForm({ ...form, vegetarian: e.target.checked })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function IngredientList() {
  const [items, setItems] = useState<api.Ingredient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.ingredientsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Recipe Id</th>
          <th>Name</th>
          <th>Amount</th>
          <th>Unit</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.recipeId}</td>
            <td>{item.name}</td>
            <td>{item.amount}</td>
            <td>{item.unit}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function IngredientForm({ onSaved }: { onSaved: (item: api.Ingredient) => void }) {
  const [form, setForm] = useState<api.NewIngredient>({ recipeId: 0, name: '', amount: 0, unit: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.ingredientsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Recipe Id
        <input type="number" value={form.recipeId} onChange={(e) => setForm({ ...form, recipeId: Number(e.target.value) })} />
      </label>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Amount
        <input type="number" value={form.amount} onChange={(e) => setForm({ ...form, amount: Number(e.target.value) })} />
      </label>
      <label>
        Unit
        <input value={form.unit} onChange={(e) => setForm({ ...form, unit: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function PlaylistList() {
  const [items, setItems] = useState<api.Playlist[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.playlistsApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Owner Id</th>
          <th>Is Public</th>
          <th>Track Count</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.ownerId}</td>
            <td>{item.isPublic ? 'Yes' : 'No'}</td>
            <td>{item.trackCount}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function PlaylistForm({ onSaved }: { onSaved: (item: api.Playlist) => void }) {
  const [form, setForm] = useState<api.NewPlaylist>({ name: '', ownerId: 0, isPublic: false, trackCount: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.playlistsApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Owner Id
        <input type="number" value={form.ownerId} onChange={(e) => setForm({ ...form, ownerId: Number(e.target.value) })} />
      </label>
      <label>
        Is Public
        <input type="checkbox" checked={form.isPublic} onChange={(e) => setForm({ ...form, isPublic: e.target.checked })} />
      </label>
      <label>
        Track Count
        <input type="number" value={form.trackCount} onChange={(e) => setForm({ ...form, trackCount: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function TrackList() {
  const [items, setItems] = useState<api.Track[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tracksApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Artist</th>
          <th>Album</th>
          <th>Duration Seconds</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.title}</td>
            <td>{item.artist}</td>
            <td>{item.album}</td>
            <td>{item.durationSeconds}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TrackForm({ onSaved }: { onSaved: (item: api.Track) => void }) {
  const [form, setForm] = useState<api.NewTrack>({ title: '', artist: '', album: '', durationSeconds: 0 });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.tracksApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Title
        <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </label>
      <label>
        Artist
        <input value={form.artist} onChange={(e) => setForm({ ...form, artist: e.target.value })} />
      </label>
      <label>
        Album
        <input value={form.album} onChange={(e) => setForm({ ...form, album: e.target.value })} />
      </label>
      <label>
        Duration Seconds
        <input type="number" value={form.durationSeconds} onChange={(e) => setForm({ ...form, durationSeconds: Number(e.target.value) })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}

export function DeviceList() {
  const [items, setItems] = useState<api.Device[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.devicesApi
      .list()
      .then((page) => setItems(page.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if (error) {
    return <p className="error">{error}</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Serial Number</th>
          <th>Firmware</th>
          <th>Online</th>
          <th>Last Seen</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id}>
            <td>{item.id}</td>
            <td>{item.name}</td>
            <td>{item.serialNumber}</td>
            <td>{item.firmware}</td>
            <td>{item.online ? 'Yes' : 'No'}</td>
            <td>{item.lastSeen}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DeviceForm({ onSaved }: { onSaved: (item: api.Device) => void }) {
  const [form, setForm] = useState<api.NewDevice>({ name: '', serialNumber: '', firmware: '', online: false, lastSeen: '' });
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const saved = await api.devicesApi.create(form);
      onSaved(saved);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <label>
        Name
        <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </label>
      <label>
        Serial Number
        <input value={form.serialNumber} onChange={(e) => setForm({ ...form, serialNumber: e.target.value })} />
      </label>
      <label>
        Firmware
        <input value={form.firmware} onChange={(e) => setForm({ ...form, firmware: e.target.value })} />
      </label>
      <label>
        Online
        <input type="checkbox" checked={form.online} onChange={(e) => setForm({ ...form, online: e.target.checked })} />
      </label>
      <label>
        Last Seen
        <input value={form.lastSeen} onChange={(e) => setForm({ ...form, lastSeen: e.target.value })} />
      </label>
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}
