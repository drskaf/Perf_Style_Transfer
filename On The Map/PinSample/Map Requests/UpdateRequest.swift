//
//  UpdateRequest.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 08/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation

struct UpdateRequest: Codable {
    let updatedAt: String
    
    enum CodingKeys: String, CodingKey {
        case updatedAt
    }
}
