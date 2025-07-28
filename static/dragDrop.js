// dragDrop.js

// Function to initialize drag-and-drop on elements with the class 'movable'
function initDragAndDrop() {
    const draggables = document.querySelectorAll('.movable');
    const container = document.getElementById('reportContent');
    
    draggables.forEach(elem => {
      elem.setAttribute('draggable', true);
      elem.addEventListener('dragstart', dragStart);
    });
  
    container.addEventListener('dragover', dragOver);
    container.addEventListener('drop', dropElement);
  }
  
  let dragSrcEl = null;
  
  function dragStart(e) {
    dragSrcEl = this;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', this.outerHTML);
    this.classList.add('dragging');
  }
  
  function dragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    return false;
  }
  
  function dropElement(e) {
    e.stopPropagation();
    if (dragSrcEl !== this) {
      // Get the container that holds the movable elements.
      const container = document.getElementById('reportContent');
      const dropY = e.clientY;
  
      // Remove the dragged element from its original position.
      dragSrcEl.parentNode.removeChild(dragSrcEl);
  
      // Insert the dragged element into its new position.
      let inserted = false;
      container.childNodes.forEach(child => {
        if (child.nodeType === Node.ELEMENT_NODE) {
          const childRect = child.getBoundingClientRect();
          if (childRect.top > dropY && !inserted) {
            container.insertBefore(dragSrcEl, child);
            inserted = true;
          }
        }
      });
      if (!inserted) {
        container.appendChild(dragSrcEl);
      }
    }
    return false;
  }
  
  // Ensure the drag-and-drop is initialized after the page loads.
  window.addEventListener('load', initDragAndDrop);
  