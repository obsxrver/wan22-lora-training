import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const utils = require('../../SillyCaption/static/video-metadata-utils.js');

class FakeNode {
  constructor() {
    this.parentNode = null;
  }

  contains(node) {
    return this === node;
  }
}

class FakeTextNode extends FakeNode {
  constructor(text) {
    super();
    this._text = String(text ?? '');
  }

  get textContent() {
    return this._text;
  }

  set textContent(value) {
    this._text = String(value ?? '');
  }
}

class FakeElement extends FakeNode {
  constructor(tagName, document) {
    super();
    this.tagName = String(tagName || '').toUpperCase();
    this.ownerDocument = document;
    this.children = [];
    this._listeners = new Map();
    this._classList = new Set();
    this._attributes = new Map();
    this._textContent = '';
    this._innerHTML = '';
    this.tabIndex = -1;
  }

  set className(value) {
    this._classList = new Set(String(value || '').split(/\s+/).filter(Boolean));
  }

  get className() {
    return Array.from(this._classList).join(' ');
  }

  get classList() {
    const self = this;
    return {
      add(...tokens) {
        tokens.flat().forEach((token) => {
          if (!token) return;
          self._classList.add(String(token));
        });
      },
      remove(...tokens) {
        tokens.flat().forEach((token) => {
          if (!token) return;
          self._classList.delete(String(token));
        });
      },
      contains(token) {
        return self._classList.has(String(token));
      },
    };
  }

  get attributes() {
    return this._attributes;
  }

  setAttribute(name, value) {
    this._attributes.set(String(name), String(value));
  }

  getAttribute(name) {
    return this._attributes.get(String(name));
  }

  appendChild(node) {
    if (!(node instanceof FakeNode)) {
      throw new TypeError('Only FakeNode instances can be appended');
    }
    node.parentNode = this;
    this.children.push(node);
    return node;
  }

  append(...nodes) {
    nodes.forEach((node) => this.appendChild(node));
  }

  addEventListener(type, handler) {
    const key = String(type);
    if (!this._listeners.has(key)) {
      this._listeners.set(key, []);
    }
    this._listeners.get(key).push(handler);
  }

  removeEventListener(type, handler) {
    const key = String(type);
    const listeners = this._listeners.get(key);
    if (!listeners) return;
    const index = listeners.indexOf(handler);
    if (index >= 0) {
      listeners.splice(index, 1);
    }
  }

  dispatchEvent(type, event = {}) {
    const key = String(type);
    const listeners = this._listeners.get(key) || [];
    const payload = {
      target: event.target || this,
      currentTarget: this,
      key: event.key,
      defaultPrevented: false,
      _stopped: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      stopPropagation() {
        this._stopped = true;
      },
    };
    if (event && typeof event === 'object') {
      for (const [prop, value] of Object.entries(event)) {
        if (prop in payload) continue;
        payload[prop] = value;
      }
    }
    for (const handler of listeners) {
      handler.call(this, payload);
      if (payload._stopped) {
        break;
      }
    }
    return !payload.defaultPrevented;
  }

  get textContent() {
    if (this.children.length === 0) {
      return this._textContent;
    }
    return this.children.map((child) => child.textContent || '').join('');
  }

  set textContent(value) {
    this.children = [];
    this._textContent = String(value ?? '');
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = String(value ?? '');
  }

  contains(node) {
    if (this === node) {
      return true;
    }
    for (const child of this.children) {
      if (child === node) {
        return true;
      }
      if (child instanceof FakeElement && child.contains(node)) {
        return true;
      }
    }
    return false;
  }
}

class FakeDocument extends FakeNode {
  constructor() {
    super();
    this._listeners = new Map();
  }

  createElement(tagName) {
    return new FakeElement(tagName, this);
  }

  createTextNode(text) {
    return new FakeTextNode(text);
  }

  addEventListener(type, handler) {
    const key = String(type);
    if (!this._listeners.has(key)) {
      this._listeners.set(key, []);
    }
    this._listeners.get(key).push(handler);
  }

  removeEventListener(type, handler) {
    const key = String(type);
    const listeners = this._listeners.get(key);
    if (!listeners) return;
    const index = listeners.indexOf(handler);
    if (index >= 0) {
      listeners.splice(index, 1);
    }
  }

  dispatchEvent(type, event = {}) {
    const key = String(type);
    const listeners = this._listeners.get(key) || [];
    const payload = {
      target: event.target || null,
      key: event.key,
      defaultPrevented: false,
      _stopped: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      stopPropagation() {
        this._stopped = true;
      },
    };
    if (event && typeof event === 'object') {
      for (const [prop, value] of Object.entries(event)) {
        if (prop in payload) continue;
        payload[prop] = value;
      }
    }
    for (const handler of listeners) {
      handler.call(this, payload);
      if (payload._stopped) {
        break;
      }
    }
    return !payload.defaultPrevented;
  }
}

function getTooltipElement(icon) {
  return icon.children.find(
    (child) => child instanceof FakeElement && child.classList.contains('video-warning-tooltip'),
  );
}

test('coarse pointer tooltips include dismissal note and respect icon sizing', () => {
  const document = new FakeDocument();
  const controller = utils.createTooltipController({
    document,
    isCoarsePointer: true,
    iconSize: 12,
  });

  const icon = controller.createWarningIcon('duration', 'Only first 5 seconds used');
  assert.ok(icon.classList.contains('video-warning--duration'));
  const tooltip = getTooltipElement(icon);
  assert.ok(tooltip, 'tooltip element should exist');
  const note = tooltip.children.find(
    (child) => child instanceof FakeElement && child.classList.contains('video-warning-tooltip-note'),
  );
  assert.ok(note, 'coarse pointer should add dismissal note');
  assert.equal(note.textContent, 'Tap again to dismiss.');
  assert.match(icon.innerHTML, /width="12"/);
});

test('click interactions toggle visibility and document clicks dismiss tooltips', () => {
  const document = new FakeDocument();
  const controller = utils.createTooltipController({
    document,
    isCoarsePointer: false,
    iconSize: 16,
  });

  const first = controller.createWarningIcon('duration', 'Only first 5 seconds used');
  const second = controller.createWarningIcon('fps', '16 FPS recommended');

  first.dispatchEvent('click', {
    preventDefault() {},
    stopPropagation() {},
  });
  assert.ok(first.classList.contains('tooltip-visible'));
  assert.strictEqual(controller.getActiveTrigger(), first);

  second.dispatchEvent('click', {
    preventDefault() {},
    stopPropagation() {},
  });
  assert.ok(second.classList.contains('tooltip-visible'));
  assert.ok(!first.classList.contains('tooltip-visible'));
  assert.strictEqual(controller.getActiveTrigger(), second);

  document.dispatchEvent('click', { target: new FakeElement('div', document) });
  assert.ok(!second.classList.contains('tooltip-visible'));
  assert.strictEqual(controller.getActiveTrigger(), null);
});

test('destroy removes active tooltip state and listeners', () => {
  const document = new FakeDocument();
  const controller = utils.createTooltipController({ document, isCoarsePointer: false });
  const icon = controller.createWarningIcon('duration', 'Only first 5 seconds used');
  icon.dispatchEvent('click', {
    preventDefault() {},
    stopPropagation() {},
  });
  assert.ok(icon.classList.contains('tooltip-visible'));
  controller.destroy();
  assert.ok(!icon.classList.contains('tooltip-visible'));
  assert.strictEqual(controller.getActiveTrigger(), null);
});
