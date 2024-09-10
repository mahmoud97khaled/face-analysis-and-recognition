criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_f.parameters(), lr=0.00005)
early_stopping_patience = 30
best_test_loss = np.inf
best_test_acc = 0
early_stopping_counter = 0

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

num_epochs = 100
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    
    train_loss, train_acc = train(model_f, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model_f, test_loader, criterion, device)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Check if testidation loss has improved
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        early_stopping_counter = 0
        torch.save(model_f.state_dict(), 'res34_res34.pth')  # Save the best model_2
    else:
        early_stopping_counter += 1
    
    # Early stopping
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break