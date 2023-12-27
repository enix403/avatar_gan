# from datasets import load_dataset as load_hgf_dataset 
# ds = load_hgf_dataset("huggingnft/cryptopunks", split="train")

torch.save({
    'generator_state': generator.state_dict(),
    'discriminator_state': discriminator.state_dict(),
},'checkpoints/001.chk')

checkpoint = torch.load('checkpoints/005.chk')
generator.load_state_dict(checkpoint['generator_state'])
discriminator.load_state_dict(checkpoint['discriminator_state'])

loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(50):
    generator.train()
    discriminator.train()
    
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    
    for real_images, _ in train_loader:
        batch_size = real_images.size(0)
        
        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))
        
        # -----------------------------
        # Train the Discriminator
        # -----------------------------
        optim_d.zero_grad()
        
        # Real images loss
        real_logits = discriminator(real_images)
        real_loss = loss_fn(real_logits, real_labels)
        
        # Fake images loss
        noise = torch.randn(batch_size, Config.noise_dim)
        fake_images = generator(noise)
        fake_logits = discriminator(fake_images.detach())
        fake_loss = loss_fn(fake_logits, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optim_d.step()
        
        # epoch_d_loss += d_loss.item()
        
        # -----------------------------
        # Train the Generator
        # -----------------------------
        optim_g.zero_grad()
        
        # Generate fake images and calculate loss
        fake_logits = discriminator(fake_images)
        g_loss = loss_fn(fake_logits, real_labels)
        g_loss.backward()
        optim_g.step()
        
        # epoch_g_loss += g_loss.item()
    