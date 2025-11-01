(function (factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory();
  } else {
    const globalObject = typeof globalThis !== 'undefined' ? globalThis : window;
    globalObject.VideoMetadataUtils = factory();
  }
})(function () {
  function formatNumber(value, digits = 2) {
    if (!Number.isFinite(value)) {
      return 'Unknown';
    }
    return Number(value)
      .toFixed(digits)
      .replace(/\.0+$/, '')
      .replace(/(\.\d*?)0+$/, '$1')
      .replace(/\.$/, '');
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) {
      return 'Unknown';
    }
    if (seconds >= 3600) {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      const secs = Math.round(seconds % 60);
      return `${hours}h ${minutes}m ${secs}s`;
    }
    if (seconds >= 60) {
      const minutes = Math.floor(seconds / 60);
      const secs = seconds % 60;
      const secsText = secs >= 10 ? Math.round(secs).toString() : secs.toFixed(1);
      return `${minutes}m ${secsText.replace(/\.0$/, '')}s`;
    }
    return seconds < 10 ? `${seconds.toFixed(2)} s` : `${seconds.toFixed(1)} s`;
  }

  function createTooltipController({ document, isCoarsePointer = false, iconSize = 16 } = {}) {
    if (!document) {
      throw new Error('document is required to create a tooltip controller');
    }

    let activeTooltipTrigger = null;
    const size = Number.isFinite(iconSize) && iconSize > 0 ? Number(iconSize) : 16;
    const iconAttributes = `width="${size}" height="${size}"`;

    const clickHandler = (event) => {
      if (!activeTooltipTrigger) {
        return;
      }
      if (activeTooltipTrigger.contains(event.target)) {
        return;
      }
      activeTooltipTrigger.classList.remove('tooltip-visible');
      activeTooltipTrigger = null;
    };

    const keydownHandler = (event) => {
      if (event.key === 'Escape' && activeTooltipTrigger) {
        activeTooltipTrigger.classList.remove('tooltip-visible');
        activeTooltipTrigger = null;
      }
    };

    document.addEventListener('click', clickHandler);
    document.addEventListener('keydown', keydownHandler);

    function createWarningIcon(type, tooltipText) {
      const wrapper = document.createElement('span');
      wrapper.className = `video-warning video-warning--${type}`;
      wrapper.setAttribute('role', 'img');
      wrapper.setAttribute('aria-label', tooltipText);
      wrapper.tabIndex = 0;
      if (type === 'duration') {
        wrapper.innerHTML =
          `<svg viewBox="0 0 24 24" ${iconAttributes} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="8"></circle><path d="M12 7v5l3 3"></path></svg>`;
      } else {
        wrapper.innerHTML =
          `<svg viewBox="0 0 24 24" ${iconAttributes} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a9 9 0 1 0-18 0"></path><path d="M12 3v3"></path><path d="M12 12l4-2"></path><path d="M5.65 5.65l2.12 2.12"></path><path d="M3 12h3"></path></svg>`;
      }

      const tooltip = document.createElement('span');
      tooltip.className = 'video-warning-tooltip';
      tooltip.appendChild(document.createTextNode(tooltipText));
      if (isCoarsePointer) {
        const note = document.createElement('span');
        note.className = 'video-warning-tooltip-note';
        note.textContent = 'Tap again to dismiss.';
        tooltip.appendChild(note);
      }
      wrapper.appendChild(tooltip);

      const showTooltip = () => {
        if (activeTooltipTrigger && activeTooltipTrigger !== wrapper) {
          activeTooltipTrigger.classList.remove('tooltip-visible');
        }
        wrapper.classList.add('tooltip-visible');
        activeTooltipTrigger = wrapper;
      };

      const hideTooltip = () => {
        if (activeTooltipTrigger === wrapper) {
          activeTooltipTrigger = null;
        }
        wrapper.classList.remove('tooltip-visible');
      };

      wrapper.addEventListener('mouseenter', showTooltip);
      wrapper.addEventListener('mouseleave', hideTooltip);
      wrapper.addEventListener('focus', showTooltip);
      wrapper.addEventListener('blur', hideTooltip);
      wrapper.addEventListener('click', (event) => {
        if (event.preventDefault) event.preventDefault();
        if (event.stopPropagation) event.stopPropagation();
        if (wrapper.classList.contains('tooltip-visible')) {
          hideTooltip();
        } else {
          showTooltip();
        }
      });
      wrapper.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          if (event.preventDefault) event.preventDefault();
          if (wrapper.classList.contains('tooltip-visible')) {
            hideTooltip();
          } else {
            showTooltip();
          }
        }
      });

      return wrapper;
    }

    function destroy() {
      document.removeEventListener('click', clickHandler);
      document.removeEventListener('keydown', keydownHandler);
      if (activeTooltipTrigger) {
        activeTooltipTrigger.classList.remove('tooltip-visible');
        activeTooltipTrigger = null;
      }
    }

    return {
      createWarningIcon,
      destroy,
      getActiveTrigger: () => activeTooltipTrigger,
      clearActiveTrigger: () => {
        if (activeTooltipTrigger) {
          activeTooltipTrigger.classList.remove('tooltip-visible');
          activeTooltipTrigger = null;
        }
      },
    };
  }

  function normalizeVideoMetadata(metadata) {
    if (!metadata || typeof metadata !== 'object') {
      return null;
    }
    const normalized = {};
    const width = Number(metadata.width);
    const height = Number(metadata.height);
    const duration = Number(metadata.duration);
    const fps = Number(metadata.fps);
    if (Number.isFinite(width) && width > 0) {
      normalized.width = width;
    }
    if (Number.isFinite(height) && height > 0) {
      normalized.height = height;
    }
    if (Number.isFinite(duration) && duration > 0) {
      normalized.duration = duration;
    }
    if (Number.isFinite(fps) && fps > 0) {
      normalized.fps = fps;
    }
    if ('frame_count' in metadata) {
      const frameCount = Number(metadata.frame_count);
      if (Number.isFinite(frameCount) && frameCount > 0) {
        normalized.frame_count = frameCount;
      }
    }
    return Object.keys(normalized).length ? normalized : null;
  }

  return {
    formatNumber,
    formatDuration,
    createTooltipController,
    normalizeVideoMetadata,
  };
});
